# export enwiki

Run the scripts to download the enwiki files in the `user/acmiyaguchi/01-export-enwiki` directory.
This should create a directory called `$HOME/scratch/trec-tot-2025/data/enwiki` with the page, pagelink, and categorylink dumps.

Change to that directory and pull the mariadb image with apptainer:

```bash
apptainer pull docker://mariadb:latest
```

Then run the `process.sbatch` script to process the enwiki files.
This is probably best run from an interactive session, but it should work in batch.

```bash
$ pwd
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki
$ tree
.
├── enwiki-20250701-categorylinks.sql.gz
├── enwiki-20250701-pagelinks.sql.gz
├── enwiki-20250701-page.sql.gz
└── mariadb_latest.sif
```

We can run the script, which will generate a few new directories to mount initialization files and output files.
We can monitor by looking at the size of the resuling `mariadb-data` directory:

```bash
watch -n 10 du -sh $HOME/scratch/trec-tot-2025/data/enwiki/mariadb-data
```

This takes an awful long time, so we've set it up to run for about 24 hours.
Hopefully it will finish in that amount of time.
