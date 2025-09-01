
# @version ^0.3.9

MAX_BPS: constant(uint256) = 10_000

name: public(String[32])
symbol: public(String[16])
decimals: public(uint8)

owner: public(address)
minter: public(address)
treasury: public(address)
treasury_bps: public(uint256)

paused: public(bool)
total_supply: public(uint256)

balances: HashMap[address, uint256]
allowances: HashMap[address, HashMap[address, uint256]]

used_block: HashMap[bytes32, bool]

event Transfer:
    sender: address
    receiver: address
    value: uint256

event Approval:
    owner: address
    spender: address
    value: uint256

event DataBlockSubmitted:
    block_hash: bytes32
    anomaly_bps: uint256
    novelty_bps: uint256
    severity: uint256
    reward: uint256
    to: address

@external
def __init__(name_: String[32], symbol_: String[16], treasury_: address, minter_: address, treasury_bps_: uint256):
    self.name = name_
    self.symbol = symbol_
    self.decimals = 18
    self.owner = msg.sender
    self.minter = minter_
    self.treasury = treasury_
    assert treasury_bps_ <= MAX_BPS
    self.treasury_bps = treasury_bps_
    self.paused = False
    self.total_supply = 0

@internal
def _only_owner():
    assert msg.sender == self.owner, "only owner"

@internal
def _only_minter():
    assert msg.sender == self.minter, "only minter"

@external
def set_minter(a: address):
    self._only_owner()
    self.minter = a

@external
def set_treasury(a: address):
    self._only_owner()
    self.treasury = a

@external
def set_treasury_bps(bps: uint256):
    self._only_owner()
    assert bps <= MAX_BPS
    self.treasury_bps = bps

@external
def pause():
    self._only_owner()
    self.paused = True

@external
def unpause():
    self._only_owner()
    self.paused = False

@view
@external
def used(blk: bytes32) -> bool:
    return self.used_block[blk]

@internal
def _mint(to: address, amount: uint256):
    assert to != empty(address)
    self.total_supply += amount
    self.balances[to] += amount
    log Transfer(empty(address), to, amount)

@internal
def _transfer(frm: address, to: address, amount: uint256):
    assert to != empty(address)
    assert self.balances[frm] >= amount
    self.balances[frm] -= amount
    self.balances[to] += amount
    log Transfer(frm, to, amount)

@external
def transfer(to: address, amount: uint256) -> bool:
    assert not self.paused
    self._transfer(msg.sender, to, amount)
    return True

@external
def approve(spender: address, amount: uint256) -> bool:
    self.allowances[msg.sender][spender] = amount
    log Approval(msg.sender, spender, amount)
    return True

@external
def transferFrom(frm: address, to: address, amount: uint256) -> bool:
    assert not self.paused
    allowed: uint256 = self.allowances[frm][msg.sender]
    assert allowed >= amount
    self.allowances[frm][msg.sender] = allowed - amount
    self._transfer(frm, to, amount)
    return True

@view
@external
def balanceOf(a: address) -> uint256:
    return self.balances[a]

@internal
def _compute_reward(anomaly_bps: uint256, novelty_bps: uint256, severity: uint256) -> uint256:
    assert anomaly_bps <= MAX_BPS
    assert novelty_bps <= MAX_BPS
    assert 1 <= severity and severity <= 5
    base: uint256 = 10 ** 18  # 1 token unit
    return base * (anomaly_bps + novelty_bps) * severity / (MAX_BPS * 5)

@external
def submit_data_block(block_hash: bytes32, anomaly_bps: uint256, novelty_bps: uint256, severity: uint256, recipient: address) -> uint256:
    self._only_minter()
    assert not self.paused
    assert recipient != empty(address)
    assert not self.used_block[block_hash], "dup"
    self.used_block[block_hash] = True

    reward: uint256 = self._compute_reward(anomaly_bps, novelty_bps, severity)
    t_fee: uint256 = reward * self.treasury_bps / MAX_BPS
    net: uint256 = reward - t_fee

    if t_fee > 0:
        self._mint(self.treasury, t_fee)
    self._mint(recipient, net)

    log DataBlockSubmitted(block_hash, anomaly_bps, novelty_bps, severity, reward, recipient)
    return reward
