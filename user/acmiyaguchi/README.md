# README

## datasets

We generate the following datasets, relative to `$HOME/trec-tot-2025/data`:

| path                                     | description                                                                                                                                                                                                                                             |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enwiki/parquet/page`                    | Page dataset from wikipedia csv dump                                                                                                                                                                                                                    |
| `enwiki/parquet/pagelinks`               | Pagelinks dataset from wikipedia csv dump                                                                                                                                                                                                               |
| `enwiki/parquet/categorylinks`           | Categorylinks dataset from wikipedia csv dump                                                                                                                                                                                                           |
| `wikidata/processed/id2wikidataid_6m/v1` | Mapping of wikipedia page ids to wikidata ids from the wikidata 6m dataset. This is generated from the json dump that wenxin put together.                                                                                                              |
| `enwiki/processed/graph/v1`              | Initial graph representation between articles using the inner-join against the wikidata 6m dataset. The page is filtered to include the main article namespace, and to remove any redirecting pages. There is a `nodes` dataset and an `edges` dataset. |

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

# copy instead
rclone copy gdrive-trec-tot-2025: $HOME/scratch/trec-tot-2025/data/gdrive
```

And I can sync the files back if needed too.
