//! Unix domain socket transport for the control protocol (DESIGN.md section 6).
//!
//! * `$XDG_RUNTIME_DIR/noiseforge/control.sock`
//! * owner-only permissions, directory included
//! * a hard size limit per message, enforced before parsing
//! * JSON lines, parsed off the realtime thread

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::protocol::{MAX_MESSAGE_BYTES, PROTOCOL_VERSION, Request, Response, encode};

/// Transport level failures.
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("protocol decode error: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("message of {size} bytes exceeds the {MAX_MESSAGE_BYTES} byte limit")]
    TooLarge { size: usize },
    #[error("protocol version {got} does not match {PROTOCOL_VERSION}")]
    VersionMismatch { got: u32 },
    #[error("the daemon closed the connection")]
    Closed,
    #[error("no daemon is listening on {0}")]
    NotListening(PathBuf),
}

/// A listening socket owned by the daemon.
pub struct Server {
    listener: UnixListener,
    path: PathBuf,
}

impl Server {
    /// Bind the socket, creating the directory and restricting permissions.
    ///
    /// A stale socket left by a crashed daemon is removed only after a
    /// connection attempt proves nobody is listening.
    pub fn bind(path: &Path) -> Result<Self, IpcError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
            let mut perms = std::fs::metadata(dir)?.permissions();
            perms.set_mode(0o700);
            std::fs::set_permissions(dir, perms)?;
        }
        if path.exists() {
            match UnixStream::connect(path) {
                Ok(_) => {
                    return Err(IpcError::Io(std::io::Error::new(
                        std::io::ErrorKind::AddrInUse,
                        format!("another daemon is already listening on {}", path.display()),
                    )));
                }
                Err(_) => {
                    // Stale socket from a previous run.
                    std::fs::remove_file(path)?;
                }
            }
        }
        let listener = UnixListener::bind(path)?;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            path: path.to_path_buf(),
        })
    }

    /// Path this server is bound to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Accept a pending connection, if any.
    pub fn accept(&self) -> Result<Option<UnixStream>, IpcError> {
        match self.listener.accept() {
            Ok((stream, _)) => {
                stream.set_read_timeout(Some(Duration::from_millis(50)))?;
                Ok(Some(stream))
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(IpcError::Io(e)),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// One client connection.
pub struct Connection {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    buffer: Vec<u8>,
    closed: bool,
    /// Set once the client asks for unsolicited telemetry.
    pub subscribed: bool,
    /// Requested push rate in Hz.
    pub push_hz: f64,
    /// Whether telemetry frames should carry the measured PSD.
    pub want_psd: bool,
    /// True when the client asked for the connection to be dropped.
    pub done: bool,
}

impl Connection {
    /// Wrap an accepted socket.
    pub fn new(stream: UnixStream) -> Result<Self, IpcError> {
        let write_half = stream.try_clone()?;
        Ok(Self {
            reader: BufReader::new(stream),
            writer: write_half,
            buffer: Vec::with_capacity(1024),
            closed: false,
            subscribed: false,
            push_hz: 15.0,
            want_psd: false,
            done: false,
        })
    }

    /// Read one request, or `Ok(None)` when nothing complete has arrived.
    pub fn poll_request(&mut self) -> Result<Option<(u64, Request)>, IpcError> {
        if self.closed {
            return Ok(None);
        }
        self.buffer.clear();
        // Read only what is already available: the daemon loop must never block
        // on a slow client.
        let mut byte = [0u8; 1];
        loop {
            match self.reader.get_mut().read(&mut byte) {
                Ok(0) => {
                    self.closed = true;
                    return Ok(None);
                }
                Ok(_) => {
                    if byte[0] == b'\n' {
                        break;
                    }
                    if self.buffer.len() >= MAX_MESSAGE_BYTES {
                        return Err(IpcError::TooLarge {
                            size: self.buffer.len(),
                        });
                    }
                    self.buffer.push(byte[0]);
                }
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    if self.buffer.is_empty() {
                        return Ok(None);
                    }
                    return Err(IpcError::Io(e));
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.closed = true;
                    return Err(IpcError::Io(e));
                }
            }
        }
        if self.buffer.is_empty() {
            return Ok(None);
        }
        if self.buffer.len() > MAX_MESSAGE_BYTES {
            return Err(IpcError::TooLarge {
                size: self.buffer.len(),
            });
        }
        // The envelope is read first so an unknown version fails cleanly.
        #[derive(serde::Deserialize)]
        struct Envelope {
            #[serde(default)]
            v: u32,
            #[serde(default)]
            id: u64,
        }
        let envelope: Envelope = serde_json::from_slice(&self.buffer)?;
        if envelope.v != PROTOCOL_VERSION {
            return Err(IpcError::VersionMismatch { got: envelope.v });
        }
        let request: Request = serde_json::from_slice(&self.buffer)?;
        Ok(Some((envelope.id, request)))
    }

    /// Send a response.
    pub fn send(&mut self, response: &Response) -> Result<(), IpcError> {
        let bytes = encode(response)?;
        if bytes.len() > MAX_MESSAGE_BYTES {
            let fallback = Response::err(
                response.id,
                format!("response of {} bytes exceeds the limit", bytes.len()),
            );
            self.writer.write_all(&encode(&fallback)?)?;
            self.writer.flush()?;
            return Ok(());
        }
        self.writer.write_all(&bytes)?;
        self.writer.flush()?;
        Ok(())
    }

    /// Whether the peer has gone away.
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

/// A blocking client used by the CLI and the GUI.
pub struct Client {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    next_id: u64,
    socket: PathBuf,
}

impl Client {
    /// Connect to a daemon socket.
    pub fn connect(path: &Path) -> Result<Self, IpcError> {
        let stream = UnixStream::connect(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound
                || e.kind() == std::io::ErrorKind::ConnectionRefused
            {
                IpcError::NotListening(path.to_path_buf())
            } else {
                IpcError::Io(e)
            }
        })?;
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        let write_half = stream.try_clone()?;
        Ok(Self {
            reader: BufReader::new(stream),
            writer: write_half,
            next_id: 1,
            socket: path.to_path_buf(),
        })
    }

    /// Socket this client is attached to.
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// Send a request and wait for its response.
    ///
    /// Unsolicited telemetry frames (id 0) that arrive in between are skipped.
    pub fn call(&mut self, request: &Request) -> Result<Response, IpcError> {
        let id = self.next_id;
        self.next_id += 1;
        let mut bytes = serde_json::to_vec(request)?;
        // Inject the envelope fields the daemon validates.
        let envelope = format!("{{\"v\":{PROTOCOL_VERSION},\"id\":{id},");
        let mut framed = Vec::with_capacity(bytes.len() + envelope.len() + 2);
        framed.extend_from_slice(envelope.as_bytes());
        // `bytes` starts with `{`, so strip it.
        if bytes.first() == Some(&b'{') {
            bytes.remove(0);
        }
        framed.extend_from_slice(&bytes);
        framed.push(b'\n');
        self.writer.write_all(&framed)?;
        self.writer.flush()?;

        loop {
            let mut line = Vec::new();
            let n = self.reader.read_until(b'\n', &mut line)?;
            if n == 0 {
                return Err(IpcError::Closed);
            }
            if line.len() > MAX_MESSAGE_BYTES {
                return Err(IpcError::TooLarge { size: line.len() });
            }
            let response: Response = match serde_json::from_slice(&line) {
                Ok(r) => r,
                Err(e) => return Err(IpcError::Decode(e)),
            };
            if response.id == id {
                return Ok(response);
            }
            // Otherwise it is a telemetry push; drop it and keep waiting.
        }
    }

    /// Read one unsolicited frame, blocking up to `timeout`.
    pub fn next_push(&mut self, timeout: Duration) -> Result<Option<Response>, IpcError> {
        self.reader.get_mut().set_read_timeout(Some(timeout))?;
        let mut line = Vec::new();
        match self.reader.read_until(b'\n', &mut line) {
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(serde_json::from_slice(&line)?)),
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                Ok(None)
            }
            Err(e) => Err(IpcError::Io(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Request;

    fn temp_socket() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "noiseforge-ipc-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        dir.join("control.sock")
    }

    #[test]
    fn socket_permissions_are_owner_only() {
        let path = temp_socket();
        let server = Server::bind(&path).unwrap();
        let mode = std::fs::metadata(server.path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "socket mode {mode:o}");
        let dir_mode = std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(dir_mode & 0o777, 0o700, "dir mode {dir_mode:o}");
        drop(server);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn bind_refuses_to_steal_a_live_socket() {
        let path = temp_socket();
        let server = Server::bind(&path).unwrap();
        let second = Server::bind(&path);
        assert!(
            second.is_err(),
            "second bind should fail while the first lives"
        );
        drop(server);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn request_response_round_trip() {
        let path = temp_socket();
        let server = Server::bind(&path).unwrap();
        let handle = std::thread::spawn({
            let path = path.clone();
            move || {
                let mut client = Client::connect(&path).unwrap();
                client.call(&Request::Play).unwrap()
            }
        });

        let stream = loop {
            if let Some(s) = server.accept().unwrap() {
                break s;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        let mut conn = Connection::new(stream).unwrap();
        let (id, request) = loop {
            if let Some(v) = conn.poll_request().unwrap() {
                break v;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        assert!(matches!(request, Request::Play));
        conn.send(&Response::empty(id)).unwrap();
        let response = handle.join().unwrap();
        assert!(response.ok);
        drop(server);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn oversized_messages_are_rejected_before_parsing() {
        let path = temp_socket();
        let server = Server::bind(&path).unwrap();
        let handle = std::thread::spawn({
            let path = path.clone();
            move || {
                let mut stream = UnixStream::connect(&path).unwrap();
                let junk = vec![b'x'; MAX_MESSAGE_BYTES + 64];
                let _ = stream.write_all(&junk);
                let _ = stream.write_all(b"\n");
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        let stream = loop {
            if let Some(s) = server.accept().unwrap() {
                break s;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        let mut conn = Connection::new(stream).unwrap();
        let mut result = Ok(None);
        for _ in 0..200 {
            result = conn.poll_request();
            if result.is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            matches!(result, Err(IpcError::TooLarge { .. })),
            "{result:?}"
        );
        handle.join().unwrap();
        drop(server);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_missing_daemon_reports_not_listening() {
        let err = Client::connect(Path::new("/nonexistent/noiseforge/control.sock"));
        assert!(matches!(err, Err(IpcError::NotListening(_))));
    }
}
