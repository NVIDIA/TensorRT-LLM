# Adding new third-party Dependencies

The markdown files in this directory contain playbooks for how to add new
third-party dependencies. Please see the document that matches the kind of
dependency you want to add:

* For C++ dependencies compiled into the extension modules via the cmake build
  and re-distributed with the wheel [see here][1]
* For python dependencies declared via wheel metadata and installed in the
  container via pip [see here][2]

[1]: cpp-thirdparty.md
[2]: py-thirdparty.md
