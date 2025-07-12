# export enwiki

Run the scripts to download the enwiki files in the `user/acmiyaguchi/01-export-enwiki` directory.
This should create a directory called `$HOME/scratch/trec-tot-2025/data/enwiki` with the page, pagelink, and categorylink dumps.

```bash
$ pwd
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki
$ tree
.
├── enwiki-20250701-categorylinks.sql.gz
├── enwiki-20250701-pagelinks.sql.gz
└── enwiki-20250701-page.sql.gz
```

Then run the `process.sbatch` script to process the enwiki files.
This will take a very long time, so we it's best to do it in a batch session
We use the `mwsql` package to iterate over the sql dump into csv.

```bash
sbatch process.sbatch page
sbatch process.sbatch pagelinks
sbatch process.sbatch categorylinks
```

```
$ ls -alh $HOME/scratch/trec-tot-2025/data/enwiki/csv/
total 66G
drwxr-xr-x. 2 amiyaguchi3 pace-ps-dsgt_clef2025 4.0K Jul 11 03:29 .
drwxr-xr-x. 6 amiyaguchi3 pace-ps-dsgt_clef2025 4.0K Jul 11 21:29 ..
-rw-r--r--. 1 amiyaguchi3 pace-ps-dsgt_clef2025  30G Jul 11 04:06 categorylinks.csv
-rw-r--r--. 1 amiyaguchi3 pace-ps-dsgt_clef2025 6.4G Jul 11 03:27 page.csv
-rw-r--r--. 1 amiyaguchi3 pace-ps-dsgt_clef2025  30G Jul 11 04:16 pagelinks.csv
```

These are some really big datasets.
These convert easily with spark.
