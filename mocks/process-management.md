# Process Management

## Inspection
```
ps aux                   # all processes
ps aux | grep nginx
top / htop               # live view
pgrep nginx              # PIDs by name
pstree                   # process tree
```

## Signals
```
kill <pid>               # SIGTERM (graceful)
kill -9 <pid>            # SIGKILL (force)
killall nginx
pkill -u alice           # kill by user
```

## Background Jobs
```
command &                # run in background
jobs                     # list bg jobs
fg %1                    # bring to foreground
bg %1                    # resume in background
nohup command &          # survive logout
```

## Priority
```
nice -n 10 command       # start with priority
renice -n 5 -p <pid>     # change running process
```

Range: -20 (highest) to 19 (lowest).
