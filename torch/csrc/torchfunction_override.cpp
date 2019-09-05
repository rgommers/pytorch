static PyObject* get_tensor_torch_function(){
	PyObject *o;
	PyObject* method = PyObject_GetAttrString((PyObject*)&THPVariableClass, "__torch_function__");
	return Py_True;
}

static PyObject *
maybe_get_attr(PyObject *obj, char *name)
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

static bool
_is_basic_python_type(PyTypeObject *tp)
{
    return (
        /* Basic number types */
        tp == &PyBool_Type ||
#if !defined(NPY_PY3K)
        tp == &PyInt_Type ||
#endif
        tp == &PyLong_Type ||
        tp == &PyFloat_Type ||
        tp == &PyComplex_Type ||

        /* Basic sequence types */
        tp == &PyList_Type ||
        tp == &PyTuple_Type ||
        tp == &PyDict_Type ||
        tp == &PySet_Type ||
        tp == &PyFrozenSet_Type ||
        tp == &PyUnicode_Type ||
        tp == &PyBytes_Type ||
/*#if !defined(NPY_PY3K)
        tp == &PyString_Type ||
#endif  DISCUSS*/

        /* other builtins */
        tp == &PySlice_Type ||
        tp == Py_TYPE(Py_None) ||
        tp == Py_TYPE(Py_Ellipsis) ||
        tp == Py_TYPE(Py_NotImplemented) ||

        /* TODO: ndarray, but we can't see PyArray_Type here */

        /* sentinel to swallow trailing || */
        false
    );
}

static PyObject*
PyArray_LookupSpecial(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* We do not need to check for special attributes on trivial types */
    if (_is_basic_python_type(tp)) {
        return NULL;
    }
    return maybe_get_attr((PyObject *)tp, name);
}

static PyObject* get_torch_function(PyObject* obj){
	static PyObject* tensor_torch_function = NULL;

	if (tensor_torch_function == NULL){
		tensor_torch_function = get_tensor_torch_function();
	}

	if(THPVariable_CheckExact(obj)){
		Py_INCREF(tensor_torch_function);
		return tensor_torch_function;
	}

	return PyArray_LookupSpecial(obj, "__torch_function__");
}

/*
 * Like list.insert(), but for C arrays of PyObject*. Skips error checking.
 */
static void
pyobject_array_insert(PyObject **array, int length, int index, PyObject *item)
{
    int j;

    for (j = length; j > index; j--) {
        array[j] = array[j - 1];
    }
    array[index] = item;
}

#define NPY_MAXARGS 32

/*
 * Collects arguments with __torch_function__ and their corresponding methods
 * in the order in which they should be tried (i.e., skipping redundant types).
 * `relevant_args` is expected to have been produced by PySequence_Fast.
 * Returns the number of arguments, or -1 on failure.
 */
static int
get_implementing_args_and_methods(PyObject *relevant_args,
                                  PyObject **implementing_args,
                                  PyObject **methods)
{
    int num_implementing_args = 0;
    Py_ssize_t i;
    int j;

    PyObject **items = PySequence_Fast_ITEMS(relevant_args);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(relevant_args);

    for (i = 0; i < length; i++) {
        int new_class = 1;
        PyObject *argument = items[i];

        /* Have we seen this type before? */
        for (j = 0; j < num_implementing_args; j++) {
            if (Py_TYPE(argument) == Py_TYPE(implementing_args[j])) {
                new_class = 0;
                break;
            }
        }
        if (new_class) {
            PyObject *method = get_torch_function(argument);

            if (method != NULL) {
                int arg_index;

                if (num_implementing_args >= NPY_MAXARGS) {
                    PyErr_Format(
                        PyExc_TypeError,
                        "maximum number (%d) of distinct argument types " \
                        "implementing __torch_function__ exceeded",
                        NPY_MAXARGS);
                    Py_DECREF(method);
                    goto fail;
                }

                /* "subclasses before superclasses, otherwise left to right" */
                arg_index = num_implementing_args;
                for (j = 0; j < num_implementing_args; j++) {
                    PyObject *other_type;
                    other_type = (PyObject *)Py_TYPE(implementing_args[j]);
                    if (PyObject_IsInstance(argument, other_type)) {
                        arg_index = j;
                        break;
                    }
                }
                Py_INCREF(argument);
                pyobject_array_insert(implementing_args, num_implementing_args,
                                      arg_index, argument);
                pyobject_array_insert(methods, num_implementing_args,
                                      arg_index, method);
                ++num_implementing_args;
            }
        }
    }
    return num_implementing_args;

fail:
    for (j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(methods[j]);
    }
    return -1;
}

/*
 * Is this object Tensor.__torch_function__?
 */
static int
is_default_array_function(PyObject *obj)
{
    static PyObject *tensor_torch_function = NULL;

    if (tensor_torch_function == NULL) {
        tensor_torch_function = get_tensor_torch_function();
    }
    return obj == tensor_torch_function;
}

#define PyUString_InternFromString PyUnicode_InternFromString
PyObject* npy_ma_str_wrapped = PyUString_InternFromString("__wrapped__");

/*
 * Core implementation of Tensor.__torch_function__. This is exposed
 * separately so we can avoid the overhead of a Python method call from
 * within `implement_array_function`.
 */
PyObject *
torch_function_method_impl(PyObject *func, PyObject *types, PyObject *args,
                           PyObject *kwargs)
{
    Py_ssize_t j;
    PyObject *implementation, *result;

    PyObject **items = PySequence_Fast_ITEMS(types);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(types);

    for (j = 0; j < length; j++) {
        int is_subclass = PyObject_IsSubclass(
            items[j], (PyObject *)&THPVariableClass);
        if (is_subclass == -1) {
            return NULL;
        }
        if (!is_subclass) {
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
    }

    implementation = PyObject_GetAttr(func, npy_ma_str_wrapped);
    if (implementation == NULL) {
        return NULL;
    }
    result = PyObject_Call(implementation, args, kwargs);
    Py_DECREF(implementation);
    return result;
}

/*
 * Calls __torch_function__ on the provided argument, with a fast-path for
 * Tensor.
 */
static PyObject *
call_torch_function(PyObject* argument, PyObject* method,
                    PyObject* public_api, PyObject* types,
                    PyObject* args, PyObject* kwargs)
{
    if (is_default_array_function(method)) {
        return torch_function_method_impl(public_api, types, args, kwargs);
    }
    else {
        return PyObject_CallFunctionObjArgs(
            method, argument, public_api, types, args, kwargs, NULL);
    }
}


/*
 * Implements the __torch_function__ protocol for a function, as described in
 * in NEP-18. See numpy.core.overrides for a full docstring.
 */
    // PyObject *NPY_UNUSED(dummy), PyObject *positional_args) discuss

PyObject *
torch_implement_torch_function(
    PyObject *dummy, PyObject *positional_args)
{
    PyObject *implementation, *public_api, *relevant_args, *args, *kwargs;

    PyObject *types = NULL;
    PyObject *implementing_args[NPY_MAXARGS];
    PyObject *torch_function_methods[NPY_MAXARGS];

    int j, any_overrides;
    int num_implementing_args = 0;
    PyObject *result = NULL;

    static PyObject *errmsg_formatter = NULL;

    if (!PyArg_UnpackTuple(
            positional_args, "implement_array_function", 5, 5,
            &implementation, &public_api, &relevant_args, &args, &kwargs)) {
        return NULL;
    }

    relevant_args = PySequence_Fast(
        relevant_args,
        "dispatcher for __torch_function__ did not return an iterable");
    if (relevant_args == NULL) {
        return NULL;
    }

    /* Collect __torch_function__ implementations */
    num_implementing_args = get_implementing_args_and_methods(
        relevant_args, implementing_args, torch_function_methods);
    if (num_implementing_args == -1) {
        goto cleanup;
    }

    /*
     * Handle the typical case of no overrides. This is merely an optimization
     * if some arguments are Tensor objects, but is also necessary if no
     * arguments implement __torch_function__ at all (e.g., if they are all
     * built-in types).
     */
    any_overrides = 0;
    for (j = 0; j < num_implementing_args; j++) {
        if (!is_default_array_function(torch_function_methods[j])) {
            any_overrides = 1;
            break;
        }
    }
    if (!any_overrides) {
        result = PyObject_Call(implementation, args, kwargs);
        goto cleanup;
    }

    /*
     * Create a Python object for types.
     * We use a tuple, because it's the fastest Python collection to create
     * and has the bonus of being immutable.
     */
    types = PyTuple_New(num_implementing_args);
    if (types == NULL) {
        goto cleanup;
    }
    for (j = 0; j < num_implementing_args; j++) {
        PyObject *arg_type = (PyObject *)Py_TYPE(implementing_args[j]);
        Py_INCREF(arg_type);
        PyTuple_SET_ITEM(types, j, arg_type);
    }

    /* Call __torch_function__ methods */
    for (j = 0; j < num_implementing_args; j++) {
        PyObject *argument = implementing_args[j];
        PyObject *method = torch_function_methods[j];

        /*
         * We use `public_api` instead of `implementation` here so
         * __torch_function__ implementations can do equality/identity
         * comparisons.
         */
        result = call_torch_function(
            argument, method, public_api, types, args, kwargs);

        if (result == Py_NotImplemented) {
            /* Try the next one */
            Py_DECREF(result);
            result = NULL;
        }
        else {
            /* Either a good result, or an exception was raised. */
            goto cleanup;
        }
    }

    /* No acceptable override found, raise TypeError. */
    // Discuss
    // npy_cache_import("numpy.core._internal",
    //                  "array_function_errmsg_formatter",
    //                  &errmsg_formatter);
    if (errmsg_formatter != NULL) {
        PyObject *errmsg = PyObject_CallFunctionObjArgs(
            errmsg_formatter, public_api, types, NULL);
        if (errmsg != NULL) {
            PyErr_SetObject(PyExc_TypeError, errmsg);
            Py_DECREF(errmsg);
        }
    }

cleanup:
    for (j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(torch_function_methods[j]);
    }
    Py_XDECREF(types);
    Py_DECREF(relevant_args);
    return result;
}

