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
drwxr-xr-x. 2 amiyaguchi3 pace-ps-dsgt_trec2025 4.0K Jul 11 03:29 .
drwxr-xr-x. 6 amiyaguchi3 pace-ps-dsgt_trec2025 4.0K Jul 11 21:29 ..
-rw-r--r--. 1 amiyaguchi3 pace-ps-dsgt_trec2025  30G Jul 11 04:06 categorylinks.csv
-rw-r--r--. 1 amiyaguchi3 pace-ps-dsgt_trec2025 6.4G Jul 11 03:27 page.csv
-rw-r--r--. 1 amiyaguchi3 pace-ps-dsgt_trec2025  30G Jul 11 04:16 pagelinks.csv
```

These are some really big datasets.
These convert easily with spark.

## resulting dataset

```
$ du -h /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki/parquet
28G     /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki/parquet
3.7G    /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki/parquet/page
11G     /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki/parquet/pagelinks
13G     /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki/parquet/categorylinks
```

### categorylinks

```
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki/parquet/page

root
 |-- cl_from: integer (nullable = true)
 |-- cl_to: string (nullable = true)
 |-- cl_sortkey: string (nullable = true)
 |-- cl_timestamp: string (nullable = true)
 |-- cl_sortkey_prefix: string (nullable = true)
 |-- cl_collation: string (nullable = true)
 |-- cl_type: string (nullable = true)
 |-- cl_collation_id: string (nullable = true)
 |-- cl_target_id: string (nullable = true)

+-------+--------------------+--------------------+-------------------+--------------------+----------------+-------+---------------+------------+
|cl_from|               cl_to|          cl_sortkey|       cl_timestamp|   cl_sortkey_prefix|    cl_collation|cl_type|cl_collation_id|cl_target_id|
+-------+--------------------+--------------------+-------------------+--------------------+----------------+-------+---------------+------------+
|  89728|Articles_with_Spa...|     *>82D*P2DrÜ\f|2025-06-16 13:20:07|                NULL|uca-default-u-kn|   page|              1|    63706583|
|  11723|Use_dmy_dates_fro...|82:D2>2D4L200Z...|2025-06-16 13:17:16|    Heineken, Freddy|uca-default-u-kn|   page|              1|    22914319|
|  91961|Pennsylvania_coun...|2@>.FRDPZH2DDN...|2025-06-16 13:20:11|                NULL|uca-default-u-kn|   page|              1|    23402361|
|  50371|          893_deaths|H8FP:FNF4.FD...|2025-06-16 13:18:51|Photios 01 Of Con...|uca-default-u-kn|   page|              1|      119386|
|  64612|American_stage_ac...|LF62LN6:D62L6...|2025-06-16 13:19:21|      Rogers, Ginger|uca-default-u-kn|   page|              1|    22915158|
+-------+--------------------+--------------------+-------------------+--------------------+----------------+-------+---------------+------------+
only showing top 5 rows
Number of rows: 205019381
```

### page

```
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki/parquet/page

root
 |-- page_id: integer (nullable = true)
 |-- page_namespace: integer (nullable = true)
 |-- page_title: string (nullable = true)
 |-- page_is_redirect: string (nullable = true)
 |-- page_is_new: string (nullable = true)
 |-- page_random: string (nullable = true)
 |-- page_touched: string (nullable = true)
 |-- page_links_updated: string (nullable = true)
 |-- page_latest: string (nullable = true)
 |-- page_len: string (nullable = true)
 |-- page_content_model: string (nullable = true)
 |-- page_lang: string (nullable = true)

+-------+--------------+--------------------+----------------+-----------+-----------------+--------------+------------------+-----------+--------+------------------+---------+
|page_id|page_namespace|          page_title|page_is_redirect|page_is_new|      page_random|  page_touched|page_links_updated|page_latest|page_len|page_content_model|page_lang|
+-------+--------------+--------------------+----------------+-----------+-----------------+--------------+------------------+-----------+--------+------------------+---------+
| 993889|             0|   Precentral_sulcus|               0|          0|   0.946535267842|20250623215021|    20250623233246| 1291870332|    2842|          wikitext|     NULL|
|1731069|             0|           Dübendorf|               0|          0|   0.626730497886|20250629125257|    20250629132003| 1286846216|   16243|          wikitext|     NULL|
|  32755|             0|Videos_and_audio_...|               0|          0|0.573170420316207|20250701054715|    20250701054840| 1294189152|   35731|          wikitext|     NULL|
| 762149|             0|                 OFI|               0|          0|   0.740687237661|20250626201941|    20250626201939| 1151818918|     709|          wikitext|     NULL|
| 305189|             0|           Pan_music|               1|          1|    0.60811002956|20250623161116|    20250602021503|   16175248|      22|          wikitext|     NULL|
+-------+--------------+--------------------+----------------+-----------+-----------------+--------------+------------------+-----------+--------+------------------+---------+
only showing top 5 rows
Number of rows: 63415160
```

### pagelinks

```
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/enwiki/parquet/pagelinks

root
 |-- pl_from: integer (nullable = true)
 |-- pl_from_namespace: integer (nullable = true)
 |-- pl_target_id: integer (nullable = true)

+--------+-----------------+------------+
| pl_from|pl_from_namespace|pl_target_id|
+--------+-----------------+------------+
|21508619|                0|         815|
|54573037|                0|        1869|
|41072278|                0|        1567|
|40166744|                0|        3525|
|28505730|                0|        3525|
+--------+-----------------+------------+
only showing top 5 rows
Number of rows: 1617680710
```
