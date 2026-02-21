# systemd Services

## Key Commands
```
systemctl start|stop|restart|reload <service>
systemctl enable|disable <service>     # boot behaviour
systemctl status <service>
journalctl -u <service> -f             # follow logs
```

## Unit File Location
Custom units: `/etc/systemd/system/<name>.service`

## Minimal Unit File
```ini
[Unit]
Description=My App

[Service]
ExecStart=/usr/bin/myapp
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Reload after editing: `systemctl daemon-reload`

## Targets (runlevels)
- `multi-user.target` → runlevel 3
- `graphical.target` → runlevel 5
