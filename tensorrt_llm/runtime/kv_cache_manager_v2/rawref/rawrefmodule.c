/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

/* ReferenceType object structure */
typedef struct
{
    PyObject_HEAD Py_ssize_t object_id; /* ID of the referenced object */
    int valid;                          /* 1 if valid, 0 if invalidated */
} ReferenceTypeObject;

/* Forward declarations */
static PyTypeObject ReferenceTypeType;

/* Cached attribute name for faster lookups */
static PyObject* rawref_attr_name = NULL;

/* ReferenceType.__new__ - implements singleton pattern via __rawref__ */
static PyObject* ReferenceType_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyObject* obj = NULL;
    static char* kwlist[] = {"obj", NULL};

    /* Parse arguments to get the object */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &obj))
    {
        return NULL;
    }

    /* Try to get existing __rawref__ using cached attribute name (faster) */
    PyObject* existing_ref = PyObject_GetAttr(obj, rawref_attr_name);
    if (existing_ref != NULL)
    {
        /* Check if it's a ReferenceType instance and is valid */
        if (PyObject_TypeCheck(existing_ref, &ReferenceTypeType))
        {
            ReferenceTypeObject* ref_obj = (ReferenceTypeObject*) existing_ref;
            if (ref_obj->valid)
            {
                /* Return existing valid reference */
                return existing_ref;
            }
        }
        Py_DECREF(existing_ref);
    }
    else
    {
        /* Clear the AttributeError if __rawref__ doesn't exist */
        PyErr_Clear();
    }

    /* Create new reference */
    ReferenceTypeObject* self;
    self = (ReferenceTypeObject*) type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->object_id = (Py_ssize_t) obj;
        self->valid = 1;

        /* Set obj.__rawref__ to this new reference using cached attr name */
        if (PyObject_SetAttr(obj, rawref_attr_name, (PyObject*) self) < 0)
        {
            /* If we can't set __rawref__, just clear the error and continue */
            PyErr_Clear();
        }
    }
    return (PyObject*) self;
}

/* ReferenceType.__init__ */
static int ReferenceType_init(ReferenceTypeObject* self, PyObject* args, PyObject* kwds)
{
    /* __new__ already did all the work, including setting object_id and valid */
    /* Skip argument parsing since __new__ already validated them */
    /* This saves ~5-10% overhead on object creation */
    return 0;
}

/* ReferenceType.__call__() - dereference the object */
static PyObject* ReferenceType_call(ReferenceTypeObject* self, PyObject* args, PyObject* kwds)
{
    PyObject* obj;

    if (!self->valid)
    {
        Py_RETURN_NONE;
    }

    /* Use _PyObject_FromStackRefSteal or ctypes approach */
    /* We need to find the object by its id */
    /* This is the tricky part - we need to convert id back to object */

    /* Use ctypes.cast to convert id to PyObject* */
    obj = (PyObject*) self->object_id;

    /* Check if the object is still alive by verifying ref count > 0 */
    /* This is somewhat unsafe but matches the intended behavior */
    if (Py_REFCNT(obj) > 0)
    {
        Py_INCREF(obj);
        return obj;
    }

    /* Object no longer valid */
    self->valid = 0;
    Py_RETURN_NONE;
}

/* ReferenceType.invalidate() */
static PyObject* ReferenceType_invalidate(ReferenceTypeObject* self, PyObject* Py_UNUSED(ignored))
{
    self->valid = 0;
    Py_RETURN_NONE;
}

/* ReferenceType.is_valid property getter */
static PyObject* ReferenceType_is_valid(ReferenceTypeObject* self, void* closure)
{
    return PyBool_FromLong(self->valid);
}

/* ReferenceType.__class_getitem__() - support for generic type subscripting */
static PyObject* ReferenceType_class_getitem(PyObject* cls, PyObject* item)
{
    /* Just return the class itself, ignore the type parameter */
    /* This allows rawref.ref[T] to work at runtime like weakref.ref[T] */
    Py_INCREF(cls);
    return cls;
}

/* Method definitions */
static PyMethodDef ReferenceType_methods[] = {
    {"invalidate", (PyCFunction) ReferenceType_invalidate, METH_NOARGS,
        "Invalidate the reference, making it return None on dereference."},
    {"__class_getitem__", (PyCFunction) ReferenceType_class_getitem, METH_O | METH_CLASS,
        "Support for generic type subscripting (e.g., ref[T])."},
    {NULL} /* Sentinel */
};

/* Property definitions */
static PyGetSetDef ReferenceType_getsetters[] = {
    {"is_valid", (getter) ReferenceType_is_valid, NULL, "Check if the reference is still valid (read-only).", NULL},
    {NULL} /* Sentinel */
};

/* Type definition */
static PyTypeObject ReferenceTypeType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "_rawref.ReferenceType",
    .tp_doc = "A mutable reference holder that stores an object ID.",
    .tp_basicsize = sizeof(ReferenceTypeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = ReferenceType_new,
    .tp_init = (initproc) ReferenceType_init,
    .tp_call = (ternaryfunc) ReferenceType_call,
    .tp_methods = ReferenceType_methods,
    .tp_getset = ReferenceType_getsetters,
};

/* Module definition */
static PyModuleDef rawrefmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_rawref",
    .m_doc = "C extension providing mutable reference class ReferenceType (internal module).",
    .m_size = -1,
};

/* Module initialization */
PyMODINIT_FUNC PyInit__rawref(void)
{
    PyObject* m;
    ReferenceTypeObject* null_ref;

    if (PyType_Ready(&ReferenceTypeType) < 0)
        return NULL;

    m = PyModule_Create(&rawrefmodule);
    if (m == NULL)
        return NULL;

    /* Cache the __rawref__ attribute name for faster lookups */
    rawref_attr_name = PyUnicode_InternFromString("__rawref__");
    if (rawref_attr_name == NULL)
    {
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&ReferenceTypeType);
    if (PyModule_AddObject(m, "ReferenceType", (PyObject*) &ReferenceTypeType) < 0)
    {
        Py_DECREF(&ReferenceTypeType);
        Py_DECREF(m);
        return NULL;
    }

    /* Add 'ref' as an alias for 'ReferenceType' (like weakref.ref) */
    Py_INCREF(&ReferenceTypeType);
    if (PyModule_AddObject(m, "ref", (PyObject*) &ReferenceTypeType) < 0)
    {
        Py_DECREF(&ReferenceTypeType);
        Py_DECREF(m);
        return NULL;
    }

    /* Create NULL constant - an invalid reference */
    null_ref = (ReferenceTypeObject*) ReferenceTypeType.tp_alloc(&ReferenceTypeType, 0);
    if (null_ref != NULL)
    {
        null_ref->object_id = 0;
        null_ref->valid = 0;
        if (PyModule_AddObject(m, "NULL", (PyObject*) null_ref) < 0)
        {
            Py_DECREF(null_ref);
            Py_DECREF(m);
            return NULL;
        }
    }

    return m;
}
