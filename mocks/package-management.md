# Package Management

## apt (Debian/Ubuntu)
```
apt update               # refresh package index
apt upgrade              # upgrade installed
apt install <pkg>
apt remove <pkg>
apt purge <pkg>          # remove + config files
apt autoremove           # remove orphans
apt search <term>
apt show <pkg>
```

## dpkg (low-level)
```
dpkg -i package.deb      # install local deb
dpkg -l | grep nginx     # list installed
dpkg -L nginx            # list installed files
dpkg --configure -a      # fix broken installs
```

## Repositories
Sources: `/etc/apt/sources.list` and `/etc/apt/sources.list.d/`

Add key: `curl -fsSL <url> | gpg --dearmor -o /usr/share/keyrings/<name>.gpg`

## Snap
```
snap install <pkg>
snap list
snap refresh
```
