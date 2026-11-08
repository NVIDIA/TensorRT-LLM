# Docs

This directory contains the stuff for building static html documentations based on [sphinx](https://www.sphinx-doc.org/en/master/).


## Build the docs
Firstly, install the sphinx:

```sh
apt-get install python3-sphinx doxygen python3-pip graphviz
```

Secondly, install the packages:

```sh
python3 -m pip install -r ./requirements.txt
```

And then, make the docs:

```sh
doxygen Doxygen # build C++ docs

make html
```

And the finally the generated html pages will locate in the `build/html` directory.


## Preview the docs locally

The basic way to preview the docs is using the `http.serve`:

```sh
cd build/html

python3 -m http.server 8081
```

And you can visit the page with your web browser with url `http://localhost:8081`.
