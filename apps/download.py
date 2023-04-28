import logging
import os
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def is_url(url, check=True):
    # Check if string is URL and check if URL exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode()
                == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    output = subprocess.check_output(['gsutil', 'du', url],
                                     shell=True,
                                     encoding='utf-8')
    if output:
        return int(output.split()[0])
    return 0


def url_getsize(url=None):
    # Return downloadable file size in bytes
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get('content-length', -1))


def curl_download(url, filename, *, silent: bool = False) -> bool:
    """Download a file from a url to a filename using curl."""
    silent_option = 'sS' if silent else ''  # silent
    proc = subprocess.run([
        'curl',
        '-#',
        f'-{silent_option}L',
        url,
        '--output',
        filename,
        '--retry',
        '9',
        '-C',
        '-',
    ])
    return proc.returncode == 0


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    file = Path(file)
    assert_msg = (f"Downloaded file '{file}' does not exist "
                  f'or size is < min_bytes={min_bytes}')
    try:  # url1
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=True)
        assert file.exists(
        ) and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        # curl download, retry and resume on fail
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            print(f'ERROR: {assert_msg}\n{error_msg}')
        print('')


def attempt_download(file, url):
    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name

        if Path(file).is_file():
            print(f'Found {url} locally at {file}')
        else:
            safe_download(file=file, url=url, min_bytes=1E5)
        return file
    return str(file)
