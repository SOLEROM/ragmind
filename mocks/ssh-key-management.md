# SSH Key Management

Generate a key pair: `ssh-keygen -t ed25519 -C "label"`

Copy public key to remote host:
```
ssh-copy-id user@host
```

Manual copy: append `~/.ssh/id_ed25519.pub` to remote `~/.ssh/authorized_keys`.

## Config file (`~/.ssh/config`)
```
Host myserver
  HostName 192.168.1.10
  User admin
  IdentityFile ~/.ssh/id_ed25519
```

## Permissions
- `~/.ssh/` → `700`
- `authorized_keys` → `600`
- Private key → `600`

Disable password auth: set `PasswordAuthentication no` in `/etc/ssh/sshd_config`.
