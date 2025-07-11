#!/bin/bash
set -e

if [ -z "$TABLE_NAME" ]; then
  echo "ERROR: TABLE_NAME environment variable not set."
  exit 1
fi

echo "INIT SCRIPT: Exporting '$TABLE_NAME' table to CSV..."

mariadb -u root -p"$MARIADB_ROOT_PASSWORD" -e "
SELECT * FROM wikidb.$TABLE_NAME INTO OUTFILE '/output/${TABLE_NAME}.csv' FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';
"
echo "INIT SCRIPT: '$TABLE_NAME' CSV export finished."
