//! Swarm event loop + a `Client` handle, the standard libp2p "command/event"
//! bridge. The event loop owns the `Swarm`; callers issue commands and await
//! results over channels, so the coordinator can write straight-line async code.

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{anyhow, Result};
use futures::StreamExt;
use libp2p::kad::{self, GetProvidersOk, QueryId, QueryResult, RecordKey};
use libp2p::request_response::{self, Message, OutboundRequestId, ResponseChannel};
use libp2p::swarm::SwarmEvent;
use libp2p::{identify, Multiaddr, PeerId, Swarm};
use tokio::sync::{mpsc, oneshot};

use hopper_net::{HopperBehaviour, HopperBehaviourEvent};
use hopper_proto::{ActivationStream, StageResponse};

/// An inbound stage request handed to the worker for processing.
pub struct InboundRequest {
    pub request: ActivationStream,
    pub channel: ResponseChannel<StageResponse>,
}

enum Command {
    Listen {
        addr: Multiaddr,
        tx: oneshot::Sender<Multiaddr>,
    },
    Dial {
        addr: Multiaddr,
        tx: oneshot::Sender<Result<(), String>>,
    },
    AddAddress {
        peer: PeerId,
        addr: Multiaddr,
    },
    Bootstrap,
    StartProviding {
        key: RecordKey,
        tx: oneshot::Sender<()>,
    },
    GetProviders {
        key: RecordKey,
        tx: oneshot::Sender<HashSet<PeerId>>,
    },
    SendStage {
        peer: PeerId,
        req: ActivationStream,
        tx: oneshot::Sender<Result<StageResponse, String>>,
    },
    Respond {
        channel: ResponseChannel<StageResponse>,
        resp: StageResponse,
    },
}

/// Cheaply-cloneable handle to drive the swarm from async code.
#[derive(Clone)]
pub struct Client {
    tx: mpsc::UnboundedSender<Command>,
}

impl Client {
    fn send(&self, cmd: Command) -> Result<()> {
        self.tx.send(cmd).map_err(|_| anyhow!("event loop closed"))
    }

    /// Start listening; resolves to the bound address.
    pub async fn listen(&self, addr: Multiaddr) -> Result<Multiaddr> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::Listen { addr, tx })?;
        Ok(rx.await?)
    }

    /// Dial a peer address (best-effort connection warm-up).
    pub async fn dial(&self, addr: Multiaddr) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::Dial { addr, tx })?;
        rx.await?.map_err(|e| anyhow!(e))
    }

    /// Add a known address for a peer to the Kademlia routing table.
    pub fn add_address(&self, peer: PeerId, addr: Multiaddr) -> Result<()> {
        self.send(Command::AddAddress { peer, addr })
    }

    /// Kick off a Kademlia bootstrap.
    pub fn bootstrap(&self) -> Result<()> {
        self.send(Command::Bootstrap)
    }

    /// Announce that this node provides `key` (a stage).
    pub async fn start_providing(&self, key: RecordKey) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::StartProviding { key, tx })?;
        Ok(rx.await?)
    }

    /// Discover providers of `key` via Kademlia.
    pub async fn get_providers(&self, key: RecordKey) -> Result<HashSet<PeerId>> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::GetProviders { key, tx })?;
        Ok(rx.await?)
    }

    /// Send a stage request to `peer` and await its response.
    pub async fn send_stage(&self, peer: PeerId, req: ActivationStream) -> Result<StageResponse> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::SendStage { peer, req, tx })?;
        rx.await?.map_err(|e| anyhow!(e))
    }

    /// Respond to an inbound stage request.
    pub fn respond(
        &self,
        channel: ResponseChannel<StageResponse>,
        resp: StageResponse,
    ) -> Result<()> {
        self.send(Command::Respond { channel, resp })
    }
}

/// Owns the swarm and translates commands ↔ swarm events.
pub struct EventLoop {
    swarm: Swarm<HopperBehaviour>,
    cmd_rx: mpsc::UnboundedReceiver<Command>,
    inbound_tx: mpsc::UnboundedSender<InboundRequest>,
    pending_send: HashMap<OutboundRequestId, oneshot::Sender<Result<StageResponse, String>>>,
    pending_providers: HashMap<QueryId, (oneshot::Sender<HashSet<PeerId>>, HashSet<PeerId>)>,
    pending_listen: VecDeque<oneshot::Sender<Multiaddr>>,
}

/// Construct the bridge around `swarm`. Returns the client handle, the event loop
/// to spawn, and the inbound-request receiver (for workers).
pub fn new(
    swarm: Swarm<HopperBehaviour>,
) -> (Client, EventLoop, mpsc::UnboundedReceiver<InboundRequest>) {
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
    let event_loop = EventLoop {
        swarm,
        cmd_rx,
        inbound_tx,
        pending_send: HashMap::new(),
        pending_providers: HashMap::new(),
        pending_listen: VecDeque::new(),
    };
    (Client { tx: cmd_tx }, event_loop, inbound_rx)
}

impl EventLoop {
    /// Run until the command channel closes.
    pub async fn run(mut self) {
        loop {
            tokio::select! {
                event = self.swarm.select_next_some() => self.on_event(event),
                cmd = self.cmd_rx.recv() => match cmd {
                    Some(cmd) => self.on_command(cmd),
                    None => return,
                },
            }
        }
    }

    fn on_event(&mut self, event: SwarmEvent<HopperBehaviourEvent>) {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                if let Some(tx) = self.pending_listen.pop_front() {
                    let _ = tx.send(address);
                }
            }
            SwarmEvent::Behaviour(HopperBehaviourEvent::RequestResponse(
                request_response::Event::Message { message, .. },
            )) => match message {
                Message::Request {
                    request, channel, ..
                } => {
                    let _ = self.inbound_tx.send(InboundRequest { request, channel });
                }
                Message::Response {
                    request_id,
                    response,
                } => {
                    if let Some(tx) = self.pending_send.remove(&request_id) {
                        let _ = tx.send(Ok(response));
                    }
                }
            },
            SwarmEvent::Behaviour(HopperBehaviourEvent::RequestResponse(
                request_response::Event::OutboundFailure {
                    request_id, error, ..
                },
            )) => {
                if let Some(tx) = self.pending_send.remove(&request_id) {
                    let _ = tx.send(Err(error.to_string()));
                }
            }
            SwarmEvent::Behaviour(HopperBehaviourEvent::Kademlia(
                kad::Event::OutboundQueryProgressed {
                    id,
                    result: QueryResult::GetProviders(Ok(ok)),
                    step,
                    ..
                },
            )) => {
                if let Some((_, acc)) = self.pending_providers.get_mut(&id) {
                    if let GetProvidersOk::FoundProviders { providers, .. } = ok {
                        acc.extend(providers);
                    }
                }
                if step.last {
                    if let Some((tx, acc)) = self.pending_providers.remove(&id) {
                        let _ = tx.send(acc);
                    }
                }
            }
            SwarmEvent::Behaviour(HopperBehaviourEvent::Identify(identify::Event::Received {
                peer_id,
                info,
                ..
            })) => {
                for addr in info.listen_addrs {
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr);
                }
            }
            _ => {}
        }
    }

    fn on_command(&mut self, cmd: Command) {
        match cmd {
            Command::Listen { addr, tx } => match self.swarm.listen_on(addr) {
                Ok(_) => self.pending_listen.push_back(tx),
                Err(_) => drop(tx),
            },
            Command::Dial { addr, tx } => {
                let _ = tx.send(self.swarm.dial(addr).map_err(|e| e.to_string()));
            }
            Command::AddAddress { peer, addr } => {
                self.swarm.behaviour_mut().kademlia.add_address(&peer, addr);
            }
            Command::Bootstrap => {
                let _ = self.swarm.behaviour_mut().kademlia.bootstrap();
            }
            Command::StartProviding { key, tx } => {
                let _ = self.swarm.behaviour_mut().kademlia.start_providing(key);
                let _ = tx.send(());
            }
            Command::GetProviders { key, tx } => {
                let id = self.swarm.behaviour_mut().kademlia.get_providers(key);
                self.pending_providers.insert(id, (tx, HashSet::new()));
            }
            Command::SendStage { peer, req, tx } => {
                let id = self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_request(&peer, req);
                self.pending_send.insert(id, tx);
            }
            Command::Respond { channel, resp } => {
                let _ = self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, resp);
            }
        }
    }
}
