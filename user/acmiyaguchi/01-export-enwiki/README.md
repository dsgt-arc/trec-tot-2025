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
