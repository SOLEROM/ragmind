# User & Group Management

## Users
```
useradd -m -s /bin/bash alice    # create with home
passwd alice                      # set password
usermod -aG sudo alice            # add to group
userdel -r alice                  # delete + home
```

## Groups
```
groupadd devs
groupdel devs
gpasswd -a alice devs
gpasswd -d alice devs
id alice                          # show uid/gid/groups
```

## Key Files
- `/etc/passwd` — user accounts
- `/etc/shadow` — hashed passwords
- `/etc/group` — group memberships

## sudo
Edit with `visudo` — never edit directly.
```
alice ALL=(ALL:ALL) ALL
%devs ALL=(ALL) NOPASSWD: /usr/bin/apt
```

`su - alice` — switch user (full login shell)
