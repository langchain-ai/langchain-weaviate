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
