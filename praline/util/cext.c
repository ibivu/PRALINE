#include <Python.h>
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

/* Traceback bit flags for the optimized pairwise aligner. */
#define TRACEBACK_MATCH_MATCH 1 << 1
#define TRACEBACK_MATCH_INSERT_UP 1 << 2
#define TRACEBACK_MATCH_INSERT_LEFT 1 << 3
#define TRACEBACK_INSERT_UP_OPEN 1 << 4
#define TRACEBACK_INSERT_UP_EXTEND 1 << 5
#define TRACEBACK_INSERT_LEFT_OPEN 1 << 6
#define TRACEBACK_INSERT_LEFT_EXTEND 1 << 7

/* Matrix offsets */
#define MATRIX_MATCH 0
#define MATRIX_INSERT_UP 1
#define MATRIX_INSERT_LEFT 2

/* Gap penalty offsets */
#define GAP_OPEN 0
#define GAP_EXTEND 1

/* Alignment modes. */
#define MODE_GLOBAL 0
#define MODE_LOCAL 1
#define MODE_SEMIGLOBAL_BOTH 2
#define MODE_SEMIGLOBAL_ONE 3
#define MODE_SEMIGLOBAL_TWO 4

static inline npy_float32
score_match_prof_prof(const npy_intp y, const npy_intp x,
                      PyArrayObject *i1, PyArrayObject *i2, PyArrayObject *i1nz, PyArrayObject *i2nz,
                      PyArrayObject *s,
                      const npy_intp i1_cols, const npy_intp i1_row_stride,
                      const npy_intp i1_col_stride, const npy_intp i2_cols,
                      const npy_intp i2_row_stride, const npy_intp i2_col_stride,
                      const npy_intp i1nz_row_stride, const npy_intp i1nz_col_stride,
                      const npy_intp i2nz_row_stride, const npy_intp i2nz_col_stride,
                      const npy_intp s_row_stride, const npy_intp s_col_stride) {

    void const *i1_data = PyArray_DATA(i1);
    void const *i2_data = PyArray_DATA(i2);
    void const *i1nz_data = PyArray_DATA(i1nz);
    void const *i2nz_data = PyArray_DATA(i2nz);
    void const *s_data = PyArray_DATA(s);

    const npy_intp i1_row_offset = i1_row_stride * y;
    const npy_intp i2_row_offset = i2_row_stride * x;
    const npy_intp i1nz_row_offset = i1nz_row_stride * y;
    const npy_intp i2nz_row_offset = i2nz_row_stride * x;

    void const *i1_base = i1_data + i1_row_offset;
    void const *i2_base = i2_data + i2_row_offset;
    void const *i1nz_base = i1nz_data + i1nz_row_offset;
    void const *i2nz_base = i2nz_data + i2nz_row_offset;

    npy_float32 score_match = 0.0;
    npy_intp n = 0;
    npy_intp i = 0;
    while(n < i1_cols) {
        const npy_intp i1nz_col_offset = i1nz_col_stride * n;
        i = *((npy_intp *)(i1nz_base + i1nz_col_offset));
        if(i < 0) {
            break;
        }
        const npy_intp i1_col_offset = i1_col_stride * i;
        const npy_intp s_row_offset = s_row_stride * i;

        npy_intp m = 0;
        npy_intp j = 0;
        while(m < i2_cols) {
            const npy_intp i2nz_col_offset = i2nz_col_stride * m;
            j = *((npy_intp *)(i2nz_base + i2nz_col_offset));
            if(j < 0) {
                break;
            }

            const npy_intp i2_col_offset = i2_col_stride * j;
            const npy_intp s_col_offset = s_col_stride * j;

            const npy_float32 p1 = *((npy_float32 *)(i1_base + i1_col_offset));
            const npy_float32 p2 = *((npy_float32 *)(i2_base + i2_col_offset));
            const npy_float32 score = *((npy_float32 *)(s_data +
                s_row_offset + s_col_offset));

            score_match += p1 * p2 * score;

            m++;
        }

        n++;
    }
    return score_match;
}

static inline PyObject *
cext_align(PyObject *self, PyObject *args, npy_uint8 mode) {
    PyArrayObject *m, *g1, *g2, *o, *t, *z;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
        &PyArray_Type, &m, &PyArray_Type, &g1, &PyArray_Type, &g2,
        &PyArray_Type, &o, &PyArray_Type, &t, &PyArray_Type, &z)) {
        return NULL;
    }

    const npy_intp g1_row_stride = PyArray_STRIDE(g1, 0);
    const npy_intp g1_col_stride = PyArray_STRIDE(g1, 1);

    const npy_intp g2_row_stride = PyArray_STRIDE(g2, 0);
    const npy_intp g2_col_stride = PyArray_STRIDE(g2, 1);

    const npy_intp o_row_stride = PyArray_STRIDE(o, 0);
    const npy_intp o_col_stride = PyArray_STRIDE(o, 1);
    const npy_intp o_mat_stride = PyArray_STRIDE(o, 2);

    const npy_intp t_row_stride = PyArray_STRIDE(t, 0);
    const npy_intp t_col_stride = PyArray_STRIDE(t, 1);
    const npy_intp t_mat_stride = PyArray_STRIDE(t, 2);

    const npy_intp z_row_stride = PyArray_STRIDE(z, 0);
    const npy_intp z_col_stride = PyArray_STRIDE(z, 1);

    const npy_intp m_row_stride = PyArray_STRIDE(m, 0);
    const npy_intp m_col_stride = PyArray_STRIDE(m, 1);
    const npy_intp m_rows = PyArray_DIM(m, 0);
    const npy_intp m_cols = PyArray_DIM(m, 1);

    /* Now loop through all cells and fill them based on the requested
     * dynamic programming algorithm. */
    for(npy_intp y = 1; y < m_rows + 1; y++) {
        const npy_intp m_row_offset = m_row_stride * (y - 1);

        for(npy_intp x = 1; x < m_cols + 1; x++) {
            const npy_intp m_col_offset = m_col_stride * (x - 1);
            const npy_uint8 has_up = y > 0;
            const npy_uint8 has_left = x > 0;
            const npy_uint8 has_upleft = has_up && has_left;
            const npy_uint8 masked = *((npy_uint8 *)(PyArray_DATA(z) +
                (z_row_stride * y) + (z_col_stride * x)));

            /* If we're either at (0, 0) or if the cell is masked
             * then leave its at the preinit value of 0.
             */
            if(!(has_up || has_left) || masked) {
                continue;
            }

            /* Calculate the scores for the insert up state. */
            npy_float32 score_insert_up_open = 0.0;
            npy_float32 score_insert_up_extend = 0.0;
            if(has_up) {
                const npy_float32 gap_open = *((npy_float32 *)(PyArray_DATA(g1) +
                    (g1_row_stride * (y-1)) + (g1_col_stride * GAP_OPEN)));
                const npy_float32 gap_extend = *((npy_float32 *)(PyArray_DATA(g1) +
                    (g1_row_stride * (y-1)) + (g1_col_stride * GAP_EXTEND)));

                score_insert_up_open = *((npy_float32 *)(PyArray_DATA(o) +
                    (o_row_stride * (y-1)) + (o_col_stride * x) +
                    (o_mat_stride * MATRIX_MATCH))) + gap_open;
                score_insert_up_extend = *((npy_float32 *)(PyArray_DATA(o) +
                    (o_row_stride * (y-1)) + (o_col_stride * x) +
                    (o_mat_stride * MATRIX_INSERT_UP))) + gap_extend;
            }

            /* Calculate the scores for the insert left state. */
            npy_float32 score_insert_left_open = 0.0;
            npy_float32 score_insert_left_extend = 0.0;
            if(has_left) {
                const npy_float32 gap_open = *((npy_float32 *)(PyArray_DATA(g2) +
                    (g2_row_stride * (x-1)) + (g2_col_stride * GAP_OPEN)));
                const npy_float32 gap_extend = *((npy_float32 *)(PyArray_DATA(g2) +
                    (g2_row_stride * (x-1)) + (g2_col_stride * GAP_EXTEND)));

                score_insert_left_open = *((npy_float32 *)(PyArray_DATA(o) +
                    (o_row_stride * y) + (o_col_stride * (x-1)) +
                    (o_mat_stride * MATRIX_MATCH))) + gap_open;
                score_insert_left_extend = *((npy_float32 *)(PyArray_DATA(o) +
                    (o_row_stride * y) + (o_col_stride * (x-1)) +
                    (o_mat_stride * MATRIX_INSERT_LEFT))) + gap_extend;
            }

            npy_float32 score_match_match = 0.0;
            npy_float32 score_match_insert_up = 0.0;
            npy_float32 score_match_insert_left = 0.0;
            if(has_upleft) {
                const npy_float32 match_score = *((npy_float32 *)
                    (PyArray_DATA(m) + m_row_offset + m_col_offset));

                score_match_match = *((npy_float32 *)(PyArray_DATA(o) +
                    (o_row_stride * (y-1)) + (o_col_stride * (x-1)) +
                    (o_mat_stride * MATRIX_MATCH))) + match_score;
                score_match_insert_up = *((npy_float32 *)(PyArray_DATA(o) +
                    (o_row_stride * (y-1)) + (o_col_stride * (x-1)) +
                    (o_mat_stride * MATRIX_INSERT_UP))) + match_score;
                score_match_insert_left = *((npy_float32 *)(PyArray_DATA(o) +
                    (o_row_stride * (y-1)) + (o_col_stride * (x-1)) +
                    (o_mat_stride * MATRIX_INSERT_LEFT))) + match_score;
            }

            /* Determine the maximum score for the max state. Add zero if we're
             * doing a local alignment. Write the result to the traceback
             * matrix.
             */
            npy_float32 score_match_max;
            if(mode == MODE_LOCAL) {
                score_match_max = 0.0;
            } else {
                score_match_max = -INFINITY;
            }
            if(has_upleft) {
                if(score_match_match > score_match_max) {
                    score_match_max = score_match_match;
                }
                if(score_match_insert_up > score_match_max) {
                    score_match_max = score_match_insert_up;
                }
                if(score_match_insert_left > score_match_max) {
                    score_match_max = score_match_insert_left;
                }

                npy_uint8 tb_match = 0;
                if(score_match_match == score_match_max) {
                    tb_match |= TRACEBACK_MATCH_MATCH;
                }
                if(score_match_insert_up == score_match_max) {
                    tb_match |= TRACEBACK_MATCH_INSERT_UP;
                }
                if(score_match_insert_left == score_match_max) {
                    tb_match |= TRACEBACK_MATCH_INSERT_LEFT;
                }

                *((npy_uint8 *)(PyArray_DATA(t) + (t_row_stride * y) +
                    (t_col_stride * x) + (t_mat_stride * MATRIX_MATCH))) =
                    tb_match;
                *((npy_float32 *)(PyArray_DATA(o) + (o_row_stride * y) +
                    (o_col_stride * x) + (o_mat_stride * MATRIX_MATCH))) =
                    score_match_max;
            }

            /* Determine the maximum score for the insert up state. Add zero
             * if we're doing a local alignment. Write the result to the
             * traceback matrix.
             */
            npy_float32 score_insert_up_max = -INFINITY;
            if(has_up) {
                if(score_insert_up_open > score_insert_up_max) {
                    score_insert_up_max = score_insert_up_open;
                }
                if(score_insert_up_extend > score_insert_up_max) {
                    score_insert_up_max = score_insert_up_extend;
                }

                npy_uint8 tb_insert_up = 0;
                if(score_insert_up_open == score_insert_up_max) {
                    tb_insert_up |= TRACEBACK_INSERT_UP_OPEN;
                }
                if(score_insert_up_extend == score_insert_up_max) {
                    tb_insert_up |= TRACEBACK_INSERT_UP_EXTEND;
                }

                *((npy_uint8 *)(PyArray_DATA(t) + (t_row_stride * y) +
                    (t_col_stride * x) + (t_mat_stride * MATRIX_INSERT_UP))) =
                    tb_insert_up;
                *((npy_float32 *)(PyArray_DATA(o) + (o_row_stride * y) +
                    (o_col_stride * x) + (o_mat_stride * MATRIX_INSERT_UP))) =
                    score_insert_up_max;
            }

            /* Determine the maximum score for the insert left state. Add zero
             * if we're doing a local alignment. Write the result to the
             * traceback matrix.
             */
            npy_float32 score_insert_left_max = -INFINITY;
            if(has_up) {
                if(score_insert_left_open > score_insert_left_max) {
                    score_insert_left_max = score_insert_left_open;
                }
                if(score_insert_left_extend > score_insert_left_max) {
                    score_insert_left_max = score_insert_left_extend;
                }

                npy_uint8 tb_insert_left = 0;
                if(score_insert_left_open == score_insert_left_max) {
                    tb_insert_left |= TRACEBACK_INSERT_LEFT_OPEN;
                }
                if(score_insert_left_extend == score_insert_left_max) {
                    tb_insert_left |= TRACEBACK_INSERT_LEFT_EXTEND;
                }

                *((npy_uint8 *)(PyArray_DATA(t) + (t_row_stride * y) +
                    (t_col_stride * x) + (t_mat_stride * MATRIX_INSERT_LEFT))) =
                    tb_insert_left;
                *((npy_float32 *)(PyArray_DATA(o) + (o_row_stride * y) +
                    (o_col_stride * x) + (o_mat_stride * MATRIX_INSERT_LEFT))) =
                    score_insert_left_max;
            }
        }
    }

    /* Since we write our results into a shared numpy array we don't
     * need to return anything. */
    Py_RETURN_NONE;
}

static PyObject *
cext_build_scores(PyObject *self, PyObject *args)
{
    PyListObject *i1s_l, *i2s_l, *i1nzs_l, *i2nzs_l, *ss_l;
    PyArrayObject *m;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", &PyList_Type, &i1s_l,
        &PyList_Type, &i2s_l, &PyList_Type, &i1nzs_l, &PyList_Type, &i2nzs_l,
        &PyList_Type, &ss_l, &PyArray_Type, &m)) {
        return NULL;
    }

    const Py_ssize_t num_sets = PyList_GET_SIZE(i1s_l);

    PyArrayObject **i1s = PyMem_New(PyArrayObject *, num_sets);
    PyArrayObject **i2s = PyMem_New(PyArrayObject *, num_sets);
    PyArrayObject **i1nzs = PyMem_New(PyArrayObject *, num_sets);
    PyArrayObject **i2nzs = PyMem_New(PyArrayObject *, num_sets);
    PyArrayObject **ss = PyMem_New(PyArrayObject *, num_sets);

    for(Py_ssize_t n = 0; n < num_sets; n++) {
        i1s[n] = (PyArrayObject *)PyList_GET_ITEM(i1s_l, n);
        i2s[n] = (PyArrayObject *)PyList_GET_ITEM(i2s_l, n);
        i1nzs[n] = (PyArrayObject *)PyList_GET_ITEM(i1nzs_l, n);
        i2nzs[n] = (PyArrayObject *)PyList_GET_ITEM(i2nzs_l, n);
        ss[n] = (PyArrayObject *)PyList_GET_ITEM(ss_l, n);
    }

    npy_intp *i1s_row_stride = PyMem_New(npy_intp, num_sets);
    npy_intp *i1s_col_stride = PyMem_New(npy_intp, num_sets);
    npy_intp *i1s_rows = PyMem_New(npy_intp, num_sets);
    npy_intp *i1s_cols = PyMem_New(npy_intp, num_sets);

    npy_intp *i2s_row_stride = PyMem_New(npy_intp, num_sets);
    npy_intp *i2s_col_stride = PyMem_New(npy_intp, num_sets);
    npy_intp *i2s_rows = PyMem_New(npy_intp, num_sets);
    npy_intp *i2s_cols = PyMem_New(npy_intp, num_sets);

    npy_intp *i1nzs_row_stride = PyMem_New(npy_intp, num_sets);
    npy_intp *i1nzs_col_stride = PyMem_New(npy_intp, num_sets);

    npy_intp *i2nzs_row_stride = PyMem_New(npy_intp, num_sets);
    npy_intp *i2nzs_col_stride = PyMem_New(npy_intp, num_sets);

    npy_intp *ss_row_stride = PyMem_New(npy_intp, num_sets);
    npy_intp *ss_col_stride = PyMem_New(npy_intp, num_sets);

    for(Py_ssize_t n = 0; n < num_sets; n++) {
        i1s_row_stride[n] = PyArray_STRIDE(i1s[n], 0);
        i1s_col_stride[n] = PyArray_STRIDE(i1s[n], 1);
        i1s_rows[n] = PyArray_DIM(i1s[n], 0);
        i1s_cols[n] = PyArray_DIM(i1s[n], 1);

        i2s_row_stride[n] = PyArray_STRIDE(i2s[n], 0);
        i2s_col_stride[n] = PyArray_STRIDE(i2s[n], 1);
        i2s_rows[n] = PyArray_DIM(i2s[n], 0);
        i2s_cols[n] = PyArray_DIM(i2s[n], 1);

        i1nzs_row_stride[n] = PyArray_STRIDE(i1nzs[n], 0);
        i1nzs_col_stride[n] = PyArray_STRIDE(i1nzs[n], 1);

        i2nzs_row_stride[n] = PyArray_STRIDE(i2nzs[n], 0);
        i2nzs_col_stride[n] = PyArray_STRIDE(i2nzs[n], 1);

        ss_row_stride[n] = PyArray_STRIDE(ss[n], 0);
        ss_col_stride[n] = PyArray_STRIDE(ss[n], 1);
    }

    const npy_intp m_row_stride = PyArray_STRIDE(m, 0);
    const npy_intp m_col_stride = PyArray_STRIDE(m, 1);

    /* Calculate pairwise match scores for all positions in both
     * sequences.
     */
    for(npy_intp y = 0; y < i1s_rows[0]; y++) {
        const npy_intp m_row_offset = m_row_stride * y;

        for(npy_intp x = 0; x < i2s_rows[0]; x++) {
            const npy_intp m_col_offset = m_col_stride * x;

            npy_float32 score = 0.0;
            for(Py_ssize_t n = 0; n < num_sets; n++) {
                PyArrayObject *i1 = i1s[n];
                PyArrayObject *i2 = i2s[n];
                PyArrayObject *i1nz = i1nzs[n];
                PyArrayObject *i2nz = i2nzs[n];
                PyArrayObject *s = ss[n];

                npy_intp i1_row_stride = i1s_row_stride[n];
                npy_intp i1_col_stride = i1s_col_stride[n];
                npy_intp i1_cols = i1s_cols[n];

                npy_intp i2_row_stride = i2s_row_stride[n];
                npy_intp i2_col_stride = i2s_col_stride[n];
                npy_intp i2_cols = i2s_cols[n];

                npy_intp i1nz_row_stride = i1nzs_row_stride[n];
                npy_intp i1nz_col_stride = i1nzs_col_stride[n];

                npy_intp i2nz_row_stride = i2nzs_row_stride[n];
                npy_intp i2nz_col_stride = i2nzs_col_stride[n];

                npy_intp s_row_stride = ss_row_stride[n];
                npy_intp s_col_stride = ss_col_stride[n];

                score += score_match_prof_prof(y, x, i1, i2, i1nz, i2nz,
                                               s, i1_cols, i1_row_stride, i1_col_stride,
                                               i2_cols, i2_row_stride,
                                               i2_col_stride, i1nz_row_stride,
                                               i1nz_col_stride, i2nz_row_stride,
                                               i2nz_col_stride,
                                               s_row_stride, s_col_stride);
            }
            *((npy_float32 *)(PyArray_DATA(m) + m_row_offset + m_col_offset)) = score;
        }
    }


    /* Deallocate memory for our temporary objects */
    PyMem_Del(i1s);
    PyMem_Del(i2s);
    PyMem_Del(i1nzs);
    PyMem_Del(i2nzs);
    PyMem_Del(ss);

    PyMem_Del(i1s_row_stride);
    PyMem_Del(i1s_col_stride);
    PyMem_Del(i1s_rows);
    PyMem_Del(i1s_cols);

    PyMem_Del(i2s_row_stride);
    PyMem_Del(i2s_col_stride);
    PyMem_Del(i2s_rows);
    PyMem_Del(i2s_cols);

    PyMem_Del(i1nzs_row_stride);
    PyMem_Del(i1nzs_col_stride);

    PyMem_Del(i2nzs_row_stride);
    PyMem_Del(i2nzs_col_stride);

    PyMem_Del(ss_row_stride);
    PyMem_Del(ss_col_stride);

    /* Since we write our results into a shared numpy array we don't
     * need to return anything. */
    Py_RETURN_NONE;
}

static PyObject *
cext_align_global(PyObject *self, PyObject *args)
{
    return cext_align(self, args, MODE_GLOBAL);
}

static PyObject *
cext_align_local(PyObject *self, PyObject *args)
{
    return cext_align(self, args, MODE_LOCAL);
}

static PyObject *
cext_align_semiglobal_both(PyObject *self, PyObject *args)
{
    return cext_align(self, args, MODE_SEMIGLOBAL_BOTH);
}

static PyObject *
cext_align_semiglobal_one(PyObject *self, PyObject *args)
{
    return cext_align(self, args, MODE_SEMIGLOBAL_ONE);
}

static PyObject *
cext_align_semiglobal_two(PyObject *self, PyObject *args)
{
    return cext_align(self, args, MODE_SEMIGLOBAL_TWO);
}

/* Tricky to get this right for both Python 2 and 3. */
struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef cext_methods[] = {
    {"cext_build_scores", (PyCFunction)cext_build_scores, METH_VARARGS,
    "Build pairwise score matrix for the DP algorithm."},
    {"cext_align_global",  (PyCFunction)cext_align_global, METH_VARARGS,
     "Global alignment"},
    {"cext_align_local",  (PyCFunction)cext_align_local, METH_VARARGS,
     "Local alignment"},
    {"cext_align_semiglobal_both",  (PyCFunction)cext_align_semiglobal_both,
    METH_VARARGS, "Semi global alignment (both sequences)"},
    {"cext_align_semiglobal_one",  (PyCFunction)cext_align_semiglobal_one,
    METH_VARARGS, "Semi global alignment (sequence one only)"},
    {"cext_align_semiglobal_two",  (PyCFunction)cext_align_semiglobal_two,
    METH_VARARGS, "Semi global alignment (sequence two only)"},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int cext_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int cext_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "cext",
        NULL,
        sizeof(struct module_state),
        cext_methods,
        NULL,
        cext_traverse,
        cext_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_cext(void)

#else
#define INITERROR return

void
initcext(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("cext", cext_methods);
#endif
    import_array();

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("cext.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

// END NEW SETUP
