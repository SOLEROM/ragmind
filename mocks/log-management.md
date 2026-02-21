# Log Management

## Key Log Locations
| Path | Purpose |
|------|---------|
| `/var/log/syslog` | General system log |
| `/var/log/auth.log` | Auth/SSH events |
| `/var/log/kern.log` | Kernel messages |
| `/var/log/dpkg.log` | Package installs |

## journald
```
journalctl -xe           # recent errors
journalctl -u nginx -f   # follow service logs
journalctl --since "1 hour ago"
journalctl -p err        # filter by priority
```

## logrotate
Config: `/etc/logrotate.d/<app>`
```
/var/log/myapp/*.log {
  daily
  rotate 7
  compress
  missingok
  notifempty
}
```

Run manually: `logrotate -f /etc/logrotate.conf`
