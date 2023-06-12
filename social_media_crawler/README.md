# Social Media Web Crawl Tool

## Description

Used to crawl Vietnamese content from multiple platforms.

Target social media platforms:

- Reddit
- Youtube
- Play Store
- App Store
- News, Newspaper
- Website


## How to install

Clone the repo by command. This repo using `python=3.9.16`

```
git clone https://github.com/shayongithub/topic_identification_zsl
```

Install required dependencies:

```
pip install -r requirements.txt
```

**Optionall**, install `pre-commit` to /.git/hooks (*if you want to contribute*)

```
pre-commit install
```

Then you can optionally run against all the files, if not the next time you commit it will automatically run

```
pre-commit run --all-files
```

## How to run

Simply enter command and give the url respective to the type of website/application you want to crawl from

```
uvicorn api:app --reload
```

### Note


- Remeber to install the **PostgreSQL** and rename the `example.env` to `.env` and add corresponding value in order to unpack the data directly into the database. You need 6 value for this

    ```
    DATABASE_PORT: The exposed port of your database
    POSTGRES_PASSWORD: The password
    POSTGRES_USER: Username
    POSTGRES_DB: Database name
    POSTGRES_HOSTNAME: The exposed ip address of your database. Normally it is localhost
    ```

- I'm running this project on my local machine using WSL2 connect to PostgresSQL database in Windows. If you are on the same page with me consider follow this [issue](https://stackoverflow.com/questions/56824788/how-to-connect-to-windows-postgres-database-from-wsl)

    + If you running on Windows, consider following this `conn_string`
        ```
        postgresql://YourUserName:YourPassword@localhost:5432/YourDatabaseName
        ```
    + if you running on Linux and want to find the hostname, consider following this [issue](https://dba.stackexchange.com/questions/217255/how-to-get-hostname-in-postgresql)
