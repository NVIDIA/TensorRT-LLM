
# TensorRT-LLM Coding Guidelines

## C++ Coding Guidelines

The TensorRT-LLM C++ Coding Guidelines are mainly derived from the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

> **General principle:** Do not fight the tooling (clang-format, clang-tidy, pre-commit) without good reason. If the tool enforces a style, follow it.

> **Note:** These guidelines have been inconsistently followed in the existing codebase. New code and modifications to existing code should adhere to these guidelines. Cleaning up existing code to match these guidelines as you touch it is encouraged, but bulk changes of surrounding code should be a separate PR.

------

#### Namespaces
1. Closing braces of namespaces should have a comment saying the namespace it closes:
```cpp
namespace foo
{
...
} // namespace foo
```
2. Anonymous namespaces use the same convention:
```cpp
namespace
{
...
} // namespace
```

#### Constants
1. Prefer `const` or `constexpr` variables over `#defines` whenever possible, as the latter are not visible to the compiler.
2. A variable that is not modified after its initialization should be declared as `const`. This applies to local and global variable declarations; it is not required for function parameters.
3. Use east-const style: place `const` to the right of the type it qualifies (e.g. `int const x` rather than `const int x`). This applies throughout, not just to constant declarations. This is enforced by clang-format (`QualifierAlignment: Right`). See [Common Pitfalls](#common-pitfalls) for how this clarifies pointer-to-const vs const-pointer.
4. Non-POD constants (e.g. `std::string`, `std::vector`, `std::unordered_map`) must not be declared at file or namespace scope, as they are subject to the [static initialization order fiasco](https://en.cppreference.com/w/cpp/language/siof). Wrap them in a function that returns a reference to a function-local static:
```cpp
// Bad: non-POD at file scope — initialization order is undefined across translation units
static std::vector<std::string> const kNames = {"foo", "bar"};

// Good: lazy initialization, safe from SIOF
std::vector<std::string> const& getNames()
{
    static std::vector<std::string> const names = {"foo", "bar"};
    return names;
}
```
   This does not apply to `constexpr`-constructible types or POD types, which are safe at file scope.
5. For naming of constants, see the Naming section of this document.

#### Literals (Recommendation)
1. Except `0` (only used in comparison for checking signedness/existence/emptiness) and `nullptr`, `true`, `false`, all other literals should only be used for variable initialization.
   Example:
```cpp
if (nbInputs == 2U){/*...*/}
```
   Should be changed to:
```cpp
constexpr size_t kNbInputsWBias = 2U;
if (nbInputs == kNbInputsWBias)
{
    /*...*/
}
```

#### Brace Notation
1. Use the [Allman indentation](https://en.wikipedia.org/wiki/Indent_style#Allman_style) style.
2. Put the semicolon for an empty `for` or `while` loop in a new line.
3. The statement forming the body of a `switch`, `while`, `do .. while` or `for` statement shall be a compound statement. (use brace-delimited statements)
4. `If` and `else` should always be followed by brace-delimited statements, even if empty or a single statement.

These rules are enforced by clang-format via the pre-commit hook.


#### Naming
1. Filenames
   * Camel case with first letter lowercase: `thisIsASubDir` and `thisIsAFilename.cpp`
   * *NOTE*: All files involved in the compilation of a compilation target (.exe/.so) must have filenames that are case-insensitive unique.

2. Types
   * All types (including, but not limited to, class names) are [camel case](https://en.wikipedia.org/wiki/Camel_case) with uppercase first letter. Example: `FooBarClass`

3. Local variables, functions, methods and namespaces
   * Camel case with first letter lowercase. Example: `localFooBar`

4. Mutable global and static variables
   * **True globals** (external linkage, visible across translation units): `g` prefix encouraged. Example: `gAllocator`
   * **File-scope** (internal linkage — `static` at file scope or in an anonymous namespace): `s` prefix encouraged. Example: `sRegistry`
   * **Function-local `static` variables**: `s` prefix optional. Example: `static std::once_flag sFlag;`
   * The `g` and `s` prefixes are encouraged but not required. All of these use camelCase.
   * **Preferred: singleton accessor pattern.** Rather than exposing a bare global or static variable, wrap it in a function that returns a reference or pointer. This provides lazy initialization and avoids static initialization order issues:
```cpp
Registry& getRegistry()
{
    // Lambda initialization is convenient when setup is non-trivial
    static auto* registry = []()
    {
        auto* r = new Registry();
        r->registerDefaults();
        return r;
    }();
    return *registry;
}
```

5. Member variables
   * **Class** (private and protected) member variables: camelCase prefixed with 'm'. Example: `mNbFooValues`.
   * **Struct** (public) member variables: camelCase with no prefix. Example: `nbFooValues`. See the [Structures and Classes](#structures-and-classes) section for when to use `struct` vs `class`.

6. Constants
   * Global constants, static constants at class-scope, and function-scope magic-number/literal constants are camelCase with prefix 'k':
```cpp
int const kDigitNum = 10;
int const kMaxBatchSize = 256;
```

> *NOTE*: Function-scope constants that are not magic numbers or literals are named like non-constant variables:
```cpp
bool const pass = a && b;
```

7. Enumerations
   * Prefer `enum class` over plain `enum`. A `using enum` declaration is acceptable to bring values into scope when needed.
   * The enum type itself is named as a type (PascalCase). Enum values follow the constant naming convention (camelCase with prefix 'k'):
```cpp
enum class ColorOption
{
    kSpecialBlue,
    kDarkRed,
};
```

8. Macros
   * See [Constants](#constants), which are preferred over `#define`.
   * If you must use macros, however, follow uppercase snakecase: `FOO_VERSION`

Notes:
* In general don't use [hungarian notation](https://en.wikipedia.org/wiki/Hungarian_notation), except for 'apps hungarian' in some cases such as 'nb' in a variable name to indicate count: `mNbLayers`. This is at the coder's discretion.
* If a constructor's parameter name `foo` conflicts with a member name `mFoo`, add a trailing underscore to the parameter name: `foo_`. This should rarely come up because class member variables are private and prefixed with `m`, and structs should prefer designated initializers or default member initializers over custom constructors.
* Literal suffixes should be upper case. For example, use `1234L` instead of `1234l`. Consider using the digit separator `'` to break up long literals for readability (e.g. `1'000'000`).


#### Tabs vs Spaces
1. Use only spaces. Do not use tabs.
2. Indent 4 spaces at a time. This is enforced automatically if you format your code using our clang-format config.


#### Formatting
1. Use the [LLVM clang-format](https://clang.llvm.org/docs/ClangFormat.html) tool for formatting your changes prior to submitting the PR. This is run automatically by the pre-commit hook and checked in CI.
2. Use a maximum of 120 characters per line. The auto formatting tool will wrap longer lines.
3. Exceptions to formatting violations must be justified on a per-case basis. Bypassing the formatting rules is discouraged, but can be achieved for exceptions as follows:
```cpp
// clang-format off
// .. Unformatted code ..
// clang-format on
```

#### Pointers and Memory Allocation
1. Use smart pointers for allocating objects on the heap.
2. When picking a smart pointer, prefer `unique_ptr` for single resource ownership and `shared_ptr` for shared resource ownership. Use `weak_ptr` only in exceptional cases.
3. Do not use smart pointers that have been deprecated in C++11.


#### Comments
1. C++ comments are required. C comments are not allowed except for special cases (inline).
2. C++ style for single-line comments. `// This is a single line comment`
3. In function calls where parameters are not obvious from inspection, use an inline C comment to document the parameter. The [`bugprone-argument-comment`](https://clang.llvm.org/extra/clang-tidy/checks/bugprone/argument-comment.html) clang-tidy check can verify these match the actual parameter names. The comment must include the `=` sign (`/*paramName=*/`); without it, the comment is ignored by the checker and name mismatches will not be caught:
```cpp
void copy(void* dst, void const* src, size_t size);

copy(/*dst=*/output, /*src=*/input, /*size=*/nbBytes);   // OK: names match, checker verifies
copy(/*src=*/output, /*dst=*/input, /*size=*/nbBytes);   // Warning: src/dst swapped
copy(/*dst*/output, /*src*/input, /*size*/nbBytes);      // No warning! Missing = means checker ignores these
```
4. If the comment is a full sentence, it should be capitalized i.e. start with capital letter and punctuated properly.
5. Follow [Doxygen rules](http://www.doxygen.nl/manual/docblocks.html) for documenting new class interfaces and function prototypes.
* For C++-style single-line comments use `//!`.
* For class members, use `//!<`.
```cpp
//! This is a Doxygen comment
//! in C++ style

struct Foo
{
    int x; //!< This is a Doxygen comment for members
};
```

#### Disabling Code
1. Use `#if` / `#endif` to disable code, preferably with a mnemonic condition like this:
```cpp
#if DEBUG_FEATURE
// ...code to be disabled...
#endif
```

```cpp
// Alternative: use a macro which evaluates to a noop in release code.
#if DEBUG_FEATURE
# define DEBUG_FEAT_CODE(x) x
#else
# define DEBUG_FEAT_CODE(x)
#endif
```

2. Avoid dead code.

```cpp
constexpr bool kDisabledFeature = false;

void foo()
{
   if (kDisabledFeature)
   {
       doSomething();
   }
}
```

3. Do NOT use comments to disable code. Use comments to explain code, not hide it.


#### Exceptions
1.  Exceptions must not be thrown across library boundaries.


#### Casts
1. Use the least forceful cast necessary, or no cast if possible, to help the compiler diagnose unintended consequences.
2. Casting a pointer to a `void*` should be implicit (except if removing `const`).
3. Casting should not remove any `const` or `volatile` qualification from the type of a pointer or reference.
4. Do not use C-style casts (other than void casts) and functional notation casts (other than explicit constructor calls).
5. Casting from a `void*` to a `T*` should be done with `static_cast`, not `reinterpret_cast`, since the latter is more forceful.
6. Use `reinterpret_cast` as a last resort, where `const_cast` and `static_cast` won't work.
7. Avoid `dynamic_cast`.


#### Expressions
1. Do not use assignment operator in subexpressions.
```cpp
// Not compliant
x = y = z;

// Not compliant
if (x = y)
{
    // ...
}
```

#### Statements
1. When practical, a `switch` statement controlled by an `enum` should have a case for each enum value and not have a default clause so that we get a compile-time error if a new enum value is added.
2. Switch statements should be well structured.  An informal guideline is to treat switch statements as structured multi-way branches and not "glorified gotos" such as:
```cpp
// Not compliant
switch (x) case 4: if (y) case 5: return 0; else default: return 1;
```
3. The "well structured" requirement prohibits fall-though except from one case label to another.   Each case clause must be terminated in a `break`, `throw`, or `return`.  If a case clause has multiple statements, the braces are optional.  The following example illustrates these requirements:
```cpp
switch (x)
{
case 0:         // Fall-through allowed from case 0: to case 1: since case 0 is empty.
case 1:
    a();
    b();
    break;
case 2:
case 4:
{              // With optional braces
    c();
    d();
    break;
}
case 5:
    c();
    throw 42;  // Terminating with throw is okay
default:
    throw 42;
}
```

4. If a switch clause is a compound statement, put the break inside the braces.
```cpp
switch (x)
{
case 0:
case 1:
{
    y();
    z();
    break;
}
...other cases...
}
```

#### Functions
1. Avoid declaring large functions as `inline`, absent a quantifiable benefit.  Remember that functions defined in class declarations are implicitly inline.
2. Rather than using the `static` keyword to mark a function as having internal linkage, prefer to use anonymous namespaces instead.
3. Every defined function must be called at least once. That is, do not have unused methods.
4. Parameter names should be consistent across function definition and corresponding function declarations.


#### Structures and Classes

The choice between `struct` and `class` is determined by whether the type has **invariants** — relationships between members that must be maintained for the object to be in a valid state.

**Use `struct`** when there are no invariants between members. All data members should be public. Any sequence of modifications to the public fields should leave the object in a valid state. Structs may have:
   * Methods that compute from the fields (e.g. `operator==`, `toString()`, `magnitude()`), as long as they do not introduce coupling between fields.
   * [Default member initializers](https://en.cppreference.com/w/cpp/language/data_members#Member_initialization) (NSDMIs) to avoid uninitialized fields.
   * Prefer [designated initializers](https://en.cppreference.com/w/cpp/language/aggregate_initialization#Designated_initializers) when constructing structs, as they make field assignments explicit.
```cpp
struct RenderConfig
{
    int width = 1920;
    int height = 1080;
    bool vsync = true;
};

auto cfg = RenderConfig{.width = 3840, .height = 2160}; // vsync defaults to true
```
   Designated initializers require the type to be an aggregate (no user-declared constructors), so prefer default member initializers over constructors when possible.

   Convenience constructors are allowed when designated initializers are insufficient — for example, a constructor that computes some fields from others by a typical rule, as long as the caller is free to modify any field afterward. A free function can serve the same purpose without losing aggregate status:
```cpp
// Preferred: free function preserves aggregate status and designated initializer support
[[nodiscard]] inline RenderConfig makeHiDpiConfig(int scaleFactor)
{
    return {.width = 1920 * scaleFactor, .height = 1080 * scaleFactor};
}

// Also acceptable: convenience constructor (but designated initializers are no longer available)
struct RenderConfig
{
    explicit RenderConfig(int scaleFactor)
        : width{1920 * scaleFactor}
        , height{1080 * scaleFactor}
    {
    }

    int width = 1920;
    int height = 1080;
    bool vsync = true;
};
```

> **Note:** If you need to cache computed results, enforce relationships between fields, or validate state after modification, the type should be a `class` with private data members.

**Use `class`** when there are invariants to maintain (e.g. `size` must equal `data.size()`, a cached hash must be recomputed when contents change). All data members should be private (or `protected` for inheritance). Provide access through `getXxx()` / `setXxx()` accessors. Member variables use the `m` prefix (see [Naming](#naming) rule 5).
```cpp
class Histogram
{
public:
    void addSample(float value);
    [[nodiscard]] float getMean() const;
    [[nodiscard]] SizeType32 getCount() const;

private:
    std::vector<float> mBins;
    float mSum = 0.0F;        // invariant: mSum == sum of all samples
    SizeType32 mCount = 0;    // invariant: mCount == total samples added
};
```

**General rules for both:**
1. Avoid dead code in templates: all class templates, function templates, and their member functions should be instantiated at least once.


#### Preprocessor Directives
1. `#define` and `#undef` of macros should be done only at global namespace. Exception: it is acceptable to `#define` a macro for a localized purpose and `#undef` it immediately after use. A common example is X-macros for generating repetitive code such as enum-to-string converters or dispatch tables:
```cpp
#define COLOR_LIST  \
    X(kRed)         \
    X(kGreen)       \
    X(kBlue)

char const* colorToString(ColorOption c)
{
    switch (c)
    {
#define X(val) case ColorOption::val: return #val;
    COLOR_LIST
#undef X
    }
}
#undef COLOR_LIST
```
2. Avoid the use of `#ifdef` and `#ifndef` directives (except in the case of header include guards). Prefer to use `#if defined(...)` or `#if !defined(...)` instead. The latter syntax is more consistent with C syntax, and allows you to use more complicated preprocessor conditionals, e.g.:
```cpp
#if defined(FOO) || defined(BAR)
void foo();
#endif // defined(FOO) || defined(BAR)
```

3. When nesting preprocessor directives, use indentation after the hash mark (#). For example:
```cpp
#if defined(FOO)
# if FOO == 0
#  define BAR 0
# elif FOO == 1
#  define BAR 5
# else
#  error "invalid FOO value"
# endif
#endif
```

4. Use `#pragma once` for header include guards. This is the convention used throughout the codebase.
   * When a traditional preprocessor guard is needed instead, the guard name must have prefix `TRTLLM_` followed by the filename, all in caps (e.g. `TRTLLM_FOO_BAR_HELLO_H` for `FooBarHello.h`). Do not use leading or trailing underscores in the guard symbol.


#### Signed vs Unsigned Integers
1. Use signed integers instead of unsigned, except for the cases below.
* The integer is a bitmap - use an unsigned type, since sign extension could lead to surprises.
* The integer is being used with an external library that expects an unsigned integer.  A common example is a loop that compares against `std::vector::size()`, such as:
```cpp
for (size_t i = 0; i < mTensors.size(); ++i) // preferred style
```
* Using only signed integers for the above would lead to prolixity and perhaps unsafe narrowing:
```cpp
for (int i = 0; i < static_cast<int>(mTensors.size()); ++i)
```
* Where possible, prefer range-based for loops or iterators to avoid the signed/unsigned issue entirely:
```cpp
for (auto const& tensor : mTensors) // preferred when index is not needed
```

#### Common Pitfalls

1. C headers should not be used directly.
   - Example: Use `<cstdint>` instead of  `<stdint.h>`
2. Do not use C library functions, whenever possible.
   * Use brace initialization or `std::fill_n()` instead of `memset()`. This is especially important when dealing with non-[POD types](https://en.cppreference.com/w/cpp/named_req/PODType). In the example below, using `memset()` will corrupt the vtable of `Foo`:
```cpp
struct Foo
{
    virtual int getX() { return x; }
    int x;
};
...

// Bad: use memset() to initialize Foo
{
    Foo foo;
    memset(&foo, 0, sizeof(foo)); // Destroys hidden virtual-function-table pointer!
}
// Good: use brace initialization to initialize Foo
{
    Foo foo = {};
}
```

3. Understand the difference between pointer-to-const and const-pointer. Using east-const style makes this clear — `const` always qualifies what is to its left:
```cpp
char const* errStr = getErrorStr(status);       // pointer to const char (pointer can be reassigned)
char const* const errStr = getErrorStr(status);  // const pointer to const char (neither can change)
char* const errStr = getErrorStr(status);        // const pointer to mutable char
```

## Appendix

####  Abbreviation Words and Compound Words as Part of Names

* Abbreviation words, which are usually fully-capitalized in literature, are treated as normal words without special capitalization, e.g. `gpuAllocator`, where GPU is converted to `gpu` before constructing the camel case name.
* Compound words, which are usually used in full in literature, e.g. `runtime`, can be abbreviated into fully capitalized letters, e.g. `RT`.

####  Terminology

* *CUDA code* is code that must be compiled with a CUDA compiler. Typically, it includes:
   * Declaration or definition of global or static variables with one of the following CUDA keywords: `__device__`, `__managed__` and `__constant__`.
   * Declaration or definition of device functions decorated with `__device__`.
   * Declaration or definition of kernels decorated with `__global__`.
   * Kernel launching with <<<...>>> syntax.

> NOTE:
   * Definition of kernel function pointer type aliases is not device code, e.g. `typedef __global__ void(*KernelFunc)(void* /*arg*/);`.
   * Definition of pointers to kernel functions is not device code, either, e.g. `__global__ void(*KernelFunc)(void* /*arg*/) = getKernelFunc(parameters);` .
   * Kernel launching with the CUDA runtime/driver API's, e.g. `cuLaunch` and `cudaLaunch`, is not CUDA code.

----

## Python Coding Guidelines
Code should adhere to [PEP 8](https://peps.python.org/pep-0008/#fn-hi), unless otherwise noted.

#### Python Standard
1. The code developed for TensorRT-LLM should conform to Python 3.8+.

#### Formatting

1. Indent code with 4 spaces.  Do not use tabs.
2. Code formatting is largely handled by the automatic tooling.  Do not override it unless it substantially improves readability.
3. Note we have "legacy" files and "new" files that are formatted by different toolchains, see <pyproject.toml>.  This results in somewhat different formatting between the two classes of files.  Most notably legacy files are 80 characters wide while new files are 100.


#### Imports
1. The linter will have opinions on import ordering.  Please follow them.
2. Do not use wildcard imports.
3. Despite the prohibition on wildcard imports, keep `__all__` updated to keep the public interface clearly documented.

#### Naming

##### Identifier Format
1. Files
- snake_case: `some_file.py`

2. Classes
- PascalCase: `class SomeClass`

3. Functions and Methods
- snake_case: `def my_awesome_function():`

4. Local Variables or Mutable Global Variables
- snake_case: `my_variable = ...`
- Single-letter variables may also be uppercase, e.g. `N`, `T`.
- Variables should not start with a number, but if you must, prefix with `k`, e.g. `k_99th_percentile = ...`

5. Constants (any scope)
- UPPER\_SNAKE\_CASE: `MY_CONSTANT = ...`

Variables and functions not part of a class’s or module’s public interface should be prefixed with an underscore.  Double underscores are permitted only if necessary to avoid name conflicts with inherited classes, and even then you should pursue alternatives.

##### Identifier Guidelines
1. Avoid shadowing variables declared in an outer scope.
2. Initialize all externally visible members of a class in the constructor.
3. For variables referencing “container” type objects that could live explicitly on the host or a GPU, e.g. referencing a Tensor, consider appending `_host` or `_device`/`_cuda` suffixes if the location is ambiguous.  Particularly if copies of the data exist in both locations.

#### Comments

1. For interfaces that may be used outside a file, prefer docstrings over comments.
2. Comments should be reserved for code within a function, or interfaces that are local to a file.
3. Avoid overcommenting.  Reserve comments for things that need explaining, or breaking up long sections of code into functional parts.  But in that case, consider helper functions.
4. For arguments to functions in the public interface to a file, documentation of Tensor-like arguments should include the expected dimensions, e.g. `[batch, seq_len, hdim]`, and the allowed dtype options if dtype is constrained.

#### Pydantic Guidelines

When defining any user-facing configuration classes (particularly `LlmArgs` or any class used in its fields), **always** use Pydantic classes rather than dataclasses or vanilla classes.

**Model Structure:**
- Inherit from `StrictBaseModel` (which sets `extra="forbid"`) to fail fast when users specify invalid field names
- Use [discriminated unions](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions) when a field needs to accept one of several possible config classes (e.g. `speculative_config` accepts any of `EagleDecodingConfig`, `MedusaDecodingConfig`, etc.)
- **Do not define `__init__` methods** - this bypasses Pydantic's validation and type coercion, and can cause subtle bugs with model inheritance. Instead:
  - For validation logic, use `@field_validator` or `@model_validator`
  - For post-validation initialization (e.g. setting up private fields and state), use [`model_post_init()`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_post_init)
  - For custom construction patterns, use classmethods (e.g. `from_yaml()`)
  - See [Defining a custom `__init__()`](https://docs.pydantic.dev/latest/concepts/models/#defining-a-custom-__init__) for more details

**Field Definitions:**
- Add descriptions to all user-facing fields via `Field(description="...")`. Avoid using comments for descriptions.
- Avoid `dict`, `object`, `Any` as field types - use properly typed alternatives
- Avoid defining mutable defaults directly; use `default_factory` instead.
  - Good: `Field(default_factory=list)`, `Field(default_factory=dict)`, `Field(default_factory=MyClass)`
  - Bad: `Field(default=[])`, `Field(default={})`, `Field(default=MyClass())` directly
- Use `Literal["value1", "value2"]` instead of `str` when a field should only accept certain values
- Prefer `PositiveInt`, `NonNegativeInt`, `NonNegativeFloat`, `PositiveFloat`, `Field(gt=0)`, `Field(ge=0)`, etc. for numeric constraints instead of defining custom validators
- Use `Field(min_length=1)` to enforce minimum length of a list

**Validation:**
- Use `@field_validator` and `@model_validator` instead of manual `validate()` or `is_valid()` methods
- Raise `ValueError` instead of using assertions
- Co-locate validation logic within the class itself rather than in a parent class, unless it depends on fields in the parent

**Serialization:**
- Avoid defining `to_dict()` methods - prefer Pydantic's built-in `model_dump()` to convert to a dictionary.
  - Good: `MyModel.model_dump()`
  - Bad: `MyModel.to_dict()`
  - Note: you can override `model_dump()` to customize its behavior, but avoid doing so unless absolutely necessary.
- Avoid defining `from_dict()` / `from_kwargs()` methods - prefer constructing the class directly from arguments.
  - Good: `MyModel(**kwargs)`, `MyModel(**my_dict)`
  - Bad: `MyModel.from_dict(kwargs)`, `MyModel.from_kwargs(kwargs)`

#### Docstring Syntax
##### Classes and Functions
Use the [Google style](https://google.github.io/styleguide/pyguide.html), which can be parsed by Sphinx.

##### Attributes and Variables
Attributes and variables can be documented inline. Attribute docstrings will be rendered under the docstring for the class. For example:
```python
class MyClass:
    """
    Description of the class.
    """
    def __init__(self):
        self.x = 5
        """<type>: Description of 'x'"""

y = 2
"""<type>: Description of 'y'"""
```

However, attribute docstrings are relatively rare and not expected.  Externally called functions should have docstrings, and their arguments should be documented.  Class initializer arguments especially should be documented.


#### Avoid Reflection
Avoid using reflection when functionality can be easily achieved without reflection.

For example, instead of:

```python
def make_complex(*args):
    x, y = args
    return dict(**locals())
```

Do:

```python
def make_complex(x, y):
    return {'x': x, 'y': y}
```

#### Error Handling
1. When using try-except blocks, limit the except to the smallest set of errors possible.

For example, instead of:

```python
try:
    open(path, "r").read()
except:
    print("Failed to open file")
```

Do:

```python
try:
    open(path, "r").read()
except FileNotFoundError:
    print("Failed to open file")
```


2. When using try-except blocks to handle multiple possible variable types (i.e. duck-typing), keep the body of the try as small as possible, using the else block to implement the logic.

For example, instead of:

```python
try:
    f.seek(0)
    f.read()
except AttributeError:
    ... # Not a file-like object, do something else
```

Do:

```python
try:
    f.seek # Do not call to minimize chance of unrelated failure
except AttributeError:
    ... # Not a file-like object, do something else
else:
    f.seek(0)
    f.read()
```

Except in exceptional circumstances, use the built-in exception types.  For which type to use when, see [https://docs.python.org/3/library/exceptions.html](https://docs.python.org/3/library/exceptions.html). Use exceptions for error handling, not return values.  And despite the example above, prefer isinstance() to duck typing where possible.

#### Static Typing

1. Static type checking at pre-commit time is opt-in by submodule PICs. This is highly recommended because static type checking eliminates an entire class of bugs and makes your code more readable and maintainable overall.
2. The presubmit system currently uses mypy.  However, many developers use pyright variants in their editors, so the code also has some `#pyright:` annotations.  As we don’t currently enforce pyright, maintaining these is best effort.  But if you notice they are broken, please fix them.
3. Do not use `typing.Any` if you can avoid it. Similarly, avoid bypassing the type checker with `# type: ignore` annotations.
4. Always annotate functions. Make the return type `None` if the function does not return anything (if you leave it empty, the type checker will infer the return type as `Any`).
5. Annotate class members and other variables when necessary. Always annotate `dataclass` and `NamedTuple` members.

```py
class Foo:
    def __init__(self, x: int) -> None:
        self.x = x  # inferred as int, no extra annotation required
        self.y: Optional[int] = None  # annotation required to prevent NoneType from being inferred
```

6.  Prefer using the built-in types `list`, `dict`, and `tuple` to the legacy `typing.List`, `typing.Dict`, and `typing.Tuple`. Similarly, use the `|` syntax instead of `typing.Union`.

```py
# Instead of
def foo(x: List[int], y: Union[int, float]) -> None:
    pass

# Do:
def foo(x: list[int], y: int | float) -> None:
    pass
```

7. Prefer specifying argument types in `Callable`s.

```py
# Type checks, but not the best style
def foo(c: Callable[..., int]) -> None:
    c(42)

# Best practice.
def foo(c: Callable[[int], int]) -> None:
    c(42)
```

8. Don't annotate variables where it is obvious/not necessary.

```py
x: int = 42 # Not required
```

9. Prefer `Literal` to `str` when a fixed set of values is expected.

```py
# Works:
def f(backend: str = "pytorch") -> None: pass

# But this is preferred:
def f(backend: Literal["pytorch", "tensorrt"] = "pytorch") -> None: pass
```

10. Use `@overload` when a return type depends on an input type. If the return type can be expressed using the input type, you can alternatively use a `TypeVar`.

```py
@overload
def foo(a: str) -> int:
    pass

@overload
def foo(a: float) -> float:
    pass

def foo(a: str | float) -> int | float:
    if isinstance(a, str):
        return 42
    return 42.0

def bar(a: float) -> None: pass

bar(foo(1.0)) # This will type check thanks to @overload

# In this example, the return type can be expressed as
T = TypeVar("T")
def baz(x: T) -> dict[str, T]:
    return {"key": x}
```

11. Use a bounded TypeVar only when the type parameter appears in both input and return positions to preserve specific type information; if it appears only in the parameters, use the bound type directly.

```py
class Foo:
    def f(self) -> None: pass

class Bar(Foo): pass

# Instead of:
# T = TypeVar("T", bound=Foo)
# def func(x: T) -> None:
#     x.f()

# We can just do:
def func(x: Foo) -> None:
    x.f()

# Here, using a bound type var is actually useful. We prevent
# func2 from losing type information.
# def func2(x: Foo) -> Foo:
#     return x
# x = func2(Bar()) # Return type is Foo

T = TypeVar("T", bound=Foo)
def func2(x: T) -> T:
    return x
x = func2(Bar()) # Return type is Bar
```

12. Use typing.Protocol for duck typing. Prefer it when
* You need an interface that third-party or unrelated classes can satisfy without inheriting from a base class.
* You want to type-check that an object has specific methods/attributes without coupling to a class hierarchy.

Do not use Protocol when a shared base class or ABC already exists and implementations naturally inherit from it — use the ABC directly. Also do not use it when you only need a union of concrete types — use Union or a type alias instead.

Note that TypeVars can also be bound to `Protocol`s. Use this feature to specify the expected interface for an argument to a generic function if duck typing is desired.

## Pre-commit Linting (Supplemental Rules)

Python files are split into two groups with separate lint toolchains:

| Group | Files | Formatting | Linting |
|-------|-------|-----------|---------|
| **A (modern)** | ~550 files | ruff format (100-char) | Full ruff rules |
| **B (legacy)** | ~1,350 files (listed in `legacy-files.txt`) | yapf (80-char) + isort + autoflake | Supplemental ruff rules, baseline-gated |

Key terminology used throughout this section:

- **Legacy files** — Python files listed in `legacy-files.txt` that haven't been migrated to the modern ruff toolchain yet.
- **Known violations** — pre-existing lint issues in legacy files, tracked in `ruff-legacy-baseline.json` (the "violation snapshot"). The snapshot records per-file, per-rule violation counts.

### Group A files

When you modify a Group A file and commit, ruff will:
1. Format the file (100-char line width)
2. Lint the **entire file** with the full rule set
3. Auto-fix what it can; report remaining issues for you to fix

### Group B (legacy) files

When you modify a Group B file and commit, the legacy tools handle formatting
(isort, yapf, autoflake) and the `ruff-legacy` hook applies supplemental lint
rules that the legacy tools don't cover (e.g., bare except, undefined names,
invalid escape sequences).

The hook is **baseline-gated** — it runs in both local pre-commit and CI:
1. Runs `ruff --fix` on staged legacy files, auto-fixing what it can
2. Compares remaining violations against the violation snapshot
3. If your change introduces a **new** violation (count exceeds the snapshot), the commit is blocked
4. Pre-existing known violations are tolerated — they won't block you

### Handling a new lint violation

If the `ruff-legacy` hook blocks your commit, fix the violation. The hook only
blocks on *new* violations your code introduced; pre-existing issues are already
accounted for in the snapshot.

### Reducing known violations

For developers who want to clean up existing tech debt in legacy files:

- **Auto-fixable violations**: `ruff check --config ruff-legacy.toml --fix` fixes the easy ones
  (unused imports, f-string conversions, comparison order, etc.)
- **Manual violations**: Many violations (bare excepts, undefined names, shadowed imports) require
  human judgment. Run `ruff check --config ruff-legacy.toml <file>` to see what remains.

After any batch cleanup, update the violation snapshot:
```bash
python scripts/legacy_utils.py lint-update-violations
```
Commit the fixed files and updated snapshot together.

The hook prints a hint when it detects your change reduced violations below the
snapshot counts, suggesting you run `--update-baseline` to tighten the ratchet.
This is informational — not blocking.

### Graduating a file from Group B to Group A

1. Remove the path from `legacy-files.txt`
2. Regenerate derived configs: `python scripts/legacy_utils.py gen-configs`
3. Update the violation snapshot: `python scripts/legacy_utils.py lint-update-violations`
4. Fix all violations under the main ruff ruleset: `ruff check --fix <file> && ruff format <file>`
5. Commit everything together (regenerated configs + snapshot + formatted file)

### Maintenance

`legacy-files.txt` is the **single source of truth** for which files are legacy. Three derived
configs are auto-generated from it — **never edit by hand**:

- `ruff-legacy.toml` (the ruff config with `include` list)
- Auto-generated blocks in `pyproject.toml` (`[tool.ruff.format]` exclude list)
- Auto-generated blocks in `.pre-commit-config.yaml` (regex file anchors)

The `verify-legacy-config` pre-commit hook catches stale configs: it regenerates expected content
from `legacy-files.txt` and diffs against the actual files. If they don't match, it fails and
tells you to run `python scripts/legacy_utils.py gen-configs`.

`ruff-legacy-baseline.json` (the violation snapshot) is a separate artifact — update it with
`python scripts/legacy_utils.py lint-update-violations`.

**Dependency chain** — after editing `legacy-files.txt`, two downstream artifacts must be updated:
```bash
# After editing legacy-files.txt (manually or via --prune):

# 1. Regenerate derived configs (ruff-legacy.toml, pyproject.toml blocks, pre-commit anchors)
python scripts/legacy_utils.py gen-configs

# 2. Update the violation snapshot (sync with the new file list)
python scripts/legacy_utils.py lint-update-violations
```

**Periodic housekeeping** — when files listed in `legacy-files.txt` are deleted or renamed by
unrelated PRs, their entries become stale. This is not fatal and doesn't break anything — no hook
matches a non-existent file — but the file list and violation snapshot can accumulate noise over
time. The generate command warns about stale entries and suggests running `prune-files`:
```bash
python scripts/legacy_utils.py prune-files          # cleans legacy-files.txt
python scripts/legacy_utils.py gen-configs           # regenerate configs
python scripts/legacy_utils.py lint-update-violations  # sync snapshot
```

**Keeping the violation snapshot current** — if your change reduces the number of known lint
violations in any tracked legacy file, it is heavily recommended (but not yet enforced) to update
the violation snapshot so the ratchet tightens. The pre-commit hook prints a hint when it detects
reductions. To update:
```bash
python scripts/legacy_utils.py lint-update-violations
```
Commit the updated `ruff-legacy-baseline.json` alongside your changes.

## Documentation Guidelines

#### CLI Options in Documentation
1. When documenting CLI commands for `trtllm-serve`, `trtllm-bench`, `trtllm-eval`, or similar tools, prefer using `--config` over `--extra_llm_api_options` for specifying configuration files.
   - `--config` is the preferred, shorter alias for configuration file options.
   - Example: `trtllm-serve --model <model_path> --config config.yaml` (preferred)
   - Avoid: `trtllm-serve --model <model_path> --extra_llm_api_options config.yaml`

## AI Coding Agent Guidance

This repository includes configuration files for AI coding agents (Claude Code, Cursor, Codex, Copilot, etc.):

- **`AGENTS.md`** — Shared project context, rules, architecture pointers, and commands. Checked into git.
- **`CLAUDE.md`** — Simple `@AGENTS.md` import indirection for claude code.
- **`CLAUDE.local.md`** — Personal developer overrides (gitignored). Create this file for your own preferences, local paths, or domain-specific context without affecting the shared config.

**Keeping `AGENTS.md` up to date**: If you change workflows, commands, architecture, or conventions that would benefit all developers and AI agents, update `AGENTS.md` in the same PR. It should evolve at the pace of the code.

## NVIDIA Copyright

1. All TensorRT-LLM Open Source Software code should contain an NVIDIA copyright header that includes the year of its latest meaningful modification.  The following block of text should be prepended to the top of all files.  This includes .cpp, .h, .cu, .py, and any other source files which are compiled or interpreted.
```cpp
/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```
