# Disk Management

## Inspect
```
lsblk                    # block devices tree
df -h                    # filesystem usage
du -sh /path/*           # directory sizes
fdisk -l                 # partition table
```

## Partition & Format
```
fdisk /dev/sdb           # interactive partitioning
mkfs.ext4 /dev/sdb1      # format
```

## Mount
```
mount /dev/sdb1 /mnt/data
umount /mnt/data
```

Persist in `/etc/fstab`:
```
/dev/sdb1  /mnt/data  ext4  defaults  0  2
```

## LVM Quick Ref
```
pvcreate /dev/sdb
vgcreate vg0 /dev/sdb
lvcreate -L 20G -n lv0 vg0
mkfs.ext4 /dev/vg0/lv0
```
