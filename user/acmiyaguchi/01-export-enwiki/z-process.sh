#!/bin/bash
set -e

echo "INIT SCRIPT: Exporting tables to CSV..."

# This script is run by the container's entrypoint.
# It automatically has access to the database using the root user.
# The MARIADB_DATABASE environment variable ensures we are using the correct database.
mariadb -u root -p"$MARIADB_ROOT_PASSWORD" -e "
SELECT * FROM wikidb.page INTO OUTFILE '/output/page.csv' FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\n';
SELECT * FROM wikidb.categorylinks INTO OUTFILE '/output/categorylinks.csv' FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\n';
SELECT * FROM wikidb.pagelinks INTO OUTFILE '/output/pagelinks.csv' FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\n';
"
echo "INIT SCRIPT: CSV export finished."