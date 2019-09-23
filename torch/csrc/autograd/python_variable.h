#pragma once

#include <torch/csrc/python_headers.h>
#include <memory>
#include <ATen/ATen.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/THP_export.h>

// Python object that backs torch.autograd.Variable
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPVariable {
    PyObject_HEAD
    // Payload
    torch::autograd::Variable cdata;
    // Hooks to be run on backwards pass (corresponds to Python attr
    // '_backwards_hooks', set by 'register_hook')
    PyObject* backward_hooks = nullptr;
};

THP_API PyObject *THPVariableClass;

bool THPVariable_initModule(PyObject *module);
THP_API PyObject * THPVariable_Wrap(torch::autograd::Variable var);

static inline bool THPVariable_CheckExact(PyObject *obj) {
  return Py_TYPE(obj) == (PyTypeObject*)THPVariableClass;
}

inline bool THPVariable_Check_Subclass(PyObject *obj){
  int is_subclass = PyObject_IsSubclass((PyObject *)Py_TYPE(obj), (PyObject *)THPVariableClass);
  if (is_subclass == -1) {
    return false;
  }
  return true;
}

inline bool THPVariable_Check(PyObject *obj)
{
  return THPVariableClass && PyObject_IsInstance(obj, THPVariableClass);
}

inline torch::autograd::Variable& THPVariable_Unpack(PyObject* obj) {
  auto var = (THPVariable*)obj;
  return var->cdata;
}

static PyObject * maybe_get_attr(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *res = (PyObject *)NULL;

    /* Attribute referenced by (char *)name */
    if (tp->tp_getattr != NULL) {
        res = (*tp->tp_getattr)(obj, name);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    /* Attribute referenced by (PyObject *)name */
    else if (tp->tp_getattro != NULL) {
        PyObject *w = PyUnicode_InternFromString(name);
        if (w == NULL) {
            return (PyObject *)NULL;
        }
        res = (*tp->tp_getattro)(obj, w);
        Py_DECREF(w);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    return res;
}

static PyObject*
PyArray_LookupSpecial(PyObject *obj, char *name)
{
  PyTypeObject *tp = Py_TYPE(obj);
  return maybe_get_attr((PyObject *)tp, name);
}

static PyObject* get_torch_function(PyObject* obj){
  return PyArray_LookupSpecial(obj, "__torch_function__");
}
