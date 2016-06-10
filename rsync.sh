rsync -avzP --update * ericmjl@rous:~/Documents/github/graph-fingerprint --exclude-from rsync_exclude.txt

rsync -avzP --update ericmjl@rous:~/Documents/github/graph-fingerprint/* ./ --exclude-from rsync_exclude.txt
