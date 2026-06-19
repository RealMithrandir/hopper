//! Real libp2p networking for Phase 3: a QUIC swarm with Kademlia stage
//! discovery and a request-response protocol carrying [`hopper_proto`] activation
//! frames. Only the activation crosses the wire (Invariant 1).
//!
//! This module provides the building blocks (codec, behaviour, swarm builder, key
//! helpers); the event loop and orchestration live in `hopper-daemon`.

use std::io;
use std::time::Duration;

use async_trait::async_trait;
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use libp2p::identity::Keypair;
use libp2p::kad::{self, store::MemoryStore};
use libp2p::request_response::{self, ProtocolSupport};
use libp2p::swarm::NetworkBehaviour;
use libp2p::{identify, StreamProtocol, Swarm};
use prost::Message;

use hopper_proto::{ActivationStream, StageResponse};

/// Request-response protocol id for streaming a stage's activation.
pub const STAGE_PROTOCOL: StreamProtocol = StreamProtocol::new("/hopper/stage/1");
/// Identify protocol id (so peers learn each other's listen addresses).
pub const IDENTIFY_PROTOCOL: &str = "/hopper/id/1";

const MAX_FRAME: usize = 16 * 1024 * 1024;

/// Kademlia provider key for `stage_id`: workers `start_providing` it, the
/// coordinator `get_providers` it.
pub fn stage_key(stage_id: usize) -> kad::RecordKey {
    kad::RecordKey::new(&format!("hopper-stage-{stage_id}"))
}

async fn read_frame<T: AsyncRead + Unpin + Send>(io: &mut T) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    io.read_exact(&mut len).await?;
    let n = u32::from_be_bytes(len) as usize;
    if n > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame too large",
        ));
    }
    let mut buf = vec![0u8; n];
    io.read_exact(&mut buf).await?;
    Ok(buf)
}

async fn write_frame<T: AsyncWrite + Unpin + Send>(io: &mut T, bytes: &[u8]) -> io::Result<()> {
    io.write_all(&(bytes.len() as u32).to_be_bytes()).await?;
    io.write_all(bytes).await?;
    io.close().await?;
    Ok(())
}

/// Length-prefixed protobuf codec for the stage request-response protocol.
#[derive(Clone, Default)]
pub struct StageCodec;

#[async_trait]
impl request_response::Codec for StageCodec {
    type Protocol = StreamProtocol;
    type Request = ActivationStream;
    type Response = StageResponse;

    async fn read_request<T>(&mut self, _: &StreamProtocol, io: &mut T) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let bytes = read_frame(io).await?;
        ActivationStream::decode(bytes.as_slice())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _: &StreamProtocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let bytes = read_frame(io).await?;
        StageResponse::decode(bytes.as_slice())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn write_request<T>(
        &mut self,
        _: &StreamProtocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_frame(io, &req.encode_to_vec()).await
    }

    async fn write_response<T>(
        &mut self,
        _: &StreamProtocol,
        io: &mut T,
        resp: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_frame(io, &resp.encode_to_vec()).await
    }
}

/// The swarm's combined behaviour: Kademlia discovery + the stage protocol +
/// identify. (`FraudProof`/gossipsub is deferred with peer-relay; see
/// `docs/TECH_SPEC.md §7.1`.)
#[derive(NetworkBehaviour)]
pub struct HopperBehaviour {
    pub kademlia: kad::Behaviour<MemoryStore>,
    pub request_response: request_response::Behaviour<StageCodec>,
    pub identify: identify::Behaviour,
}

impl HopperBehaviour {
    /// Construct the behaviour for a node with the given keypair.
    pub fn new(key: &Keypair) -> Self {
        let peer_id = key.public().to_peer_id();
        let mut kademlia = kad::Behaviour::new(peer_id, MemoryStore::new(peer_id));
        // Force server mode: on localhost there is no external-address detection
        // to auto-promote us, but we must store/serve provider records for stage
        // discovery to work.
        kademlia.set_mode(Some(kad::Mode::Server));
        let request_response = request_response::Behaviour::with_codec(
            StageCodec,
            [(STAGE_PROTOCOL, ProtocolSupport::Full)],
            request_response::Config::default(),
        );
        let identify = identify::Behaviour::new(identify::Config::new(
            IDENTIFY_PROTOCOL.into(),
            key.public(),
        ));
        Self {
            kademlia,
            request_response,
            identify,
        }
    }
}

/// Build a tokio QUIC swarm running [`HopperBehaviour`].
pub fn build_swarm(
    keypair: Keypair,
) -> Result<Swarm<HopperBehaviour>, Box<dyn std::error::Error + Send + Sync>> {
    let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_quic()
        .with_behaviour(HopperBehaviour::new)?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();
    Ok(swarm)
}
