import subprocess


def area(r):
    # if DEBUG:
    #   print("Computing area of %r" % r)
    return r.length * r.width


CONFIG_FILE = "foo.txt"


def process_request(request):
    password = request.GET["password"]

    # BAD: Inbound authentication made by comparison to string literal
    if password == "myPa55word":
        redirect("login")

    hashed_password = load_from_config("hashed_password", CONFIG_FILE)
    salt = load_from_config("salt", CONFIG_FILE)

    print("Hashed password: %r" % hashed_password)
    print("Salt: %r" % salt)


def redirect(url):
    # TODO: Implement redirect logic
    pass


def load_from_config(key, config_file):
    # TODO: Implement loading from config logic
    pass

# https://codeql.github.com/codeql-query-help/python/py-side-effect-in-assert/
assert subprocess.call(['run-backup']) == 0


#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Desc   ：Log Injection
"""
from flask import Flask
from flask import request
from django.utils.log import request_logger
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/bad1')
def bad1():
    name = request.args.get('name')
    app.logger.info('User name: ' + name) # Bad
    return 'bad1'

@app.route('/bad2')
def bad2():
    name = request.args.get('name')
    logging.info('User name: ' + name) # Bad
    return 'bad2'

@app.route('/bad3')
def bad3():
    name = request.args.get('name')
    request_logger.warn('User name: ' + name) # Bad
    return 'bad3'

@app.route('/bad4')
def bad4():
    name = request.args.get('name')
    logtest = logging.getLogger('test')
    logtest.debug('User name: ' + name) # Bad
    return 'bad4'

if __name__ == '__main__':
    app.debug = True
    handler = logging.FileHandler('log')
    app.logger.addHandler(handler)
    app.run()


from flask import Flask, request, redirect
import re

app = Flask(__name__)

UNSAFE_REGEX = re.compile("(www|beta).example.com/")
SAFE_REGEX = re.compile(r"(www|beta)\.example\.com/")

@app.route('/some/path/bad')
def unsafe(request):
    target = request.args.get('target', '')
    if UNSAFE_REGEX.match(target):
        return redirect(target)