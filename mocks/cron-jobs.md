# Cron Jobs

Edit user crontab: `crontab -e`
List crontab: `crontab -l`
System-wide: `/etc/crontab`, `/etc/cron.d/`

## Cron Syntax
```
*  *  *  *  *  command
│  │  │  │  └─ day of week (0–7, Sun=0 or 7)
│  │  │  └──── month (1–12)
│  │  └─────── day of month (1–31)
│  └────────── hour (0–23)
└───────────── minute (0–59)
```

## Examples
```
0 2 * * *   /usr/bin/backup.sh        # daily at 02:00
*/15 * * * * /usr/bin/check.sh        # every 15 min
0 0 1 * *   /usr/bin/monthly.sh       # first of month
```

Logs: `grep CRON /var/log/syslog`
