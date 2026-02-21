# Network Troubleshooting

## Connectivity
```
ping -c 4 8.8.8.8
traceroute 8.8.8.8
mtr 8.8.8.8              # continuous traceroute
curl -I https://example.com
```

## Interfaces & Routes
```
ip addr show
ip link set eth0 up/down
ip route show
ip route add default via 192.168.1.1
```

## DNS
```
dig example.com
dig @8.8.8.8 example.com
nslookup example.com
resolvectl status        # systemd-resolved
```

## Ports & Sockets
```
ss -tulnp                # listening ports
ss -s                    # socket summary
netstat -tulnp           # older alternative
```

## Packet Capture
```
tcpdump -i eth0 port 80 -w capture.pcap
```
