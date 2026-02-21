# UFW Firewall

## Setup
```
ufw enable
ufw disable
ufw status verbose
ufw reset              # clear all rules
```

## Rules
```
ufw allow 22/tcp
ufw allow 80
ufw allow from 192.168.1.0/24
ufw deny 3306
ufw delete allow 80
```

## Application Profiles
```
ufw app list
ufw allow 'Nginx Full'
```

## Default Policies
```
ufw default deny incoming
ufw default allow outgoing
```

Logs: `ufw logging on` then check `/var/log/ufw.log`

> iptables is the underlying engine; `iptables -L -n -v` shows raw rules.
