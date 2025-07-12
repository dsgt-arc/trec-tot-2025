# README

## notes

### Downloading wikipedia data

We are going to download the following files:

- https://dumps.wikimedia.org/enwiki/20250701/
-     enwiki-20250701-page.sql.gz 2.2 GB
-     enwiki-20250701-categorylinks.sql.gz 3.7 GB
-     enwiki-20250701-pagelinks.sql.gz 6.4 GB

We will then need to read these files into mysql or mariadb, and export them into parquet for further analysis.
The best way to do this is with aria2c.

```bash
uv tool install aria2c

cd download
./download-enwiki.sbatch
```

### syncing from google drive

I've configured rclone to sync directories to the shared google drive for the project.

```bash
rclone sync gdrive-trec-tot-2025: $HOME/scratch/trec-tot-2025/data/gdrive
```

And I can sync the files back if needed too.
