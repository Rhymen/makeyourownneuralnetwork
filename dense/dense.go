package dense

import (
	"fmt"
	"os"
	"math/rand"
	"strconv"
)

type Matrix [][]float64

func Zeros(rows, columns int) *Matrix {
	out := make(Matrix, rows)

	for i := range out {
		out[i] = make([]float64, columns)
	}

	return &out
}

func New(rows, columns int) func(elements ...float64) *Matrix {
	out := Zeros(rows, columns)

	return func(elements ...float64) *Matrix {
		if len(elements) != rows*columns {
			fmt.Fprintf(os.Stderr, "len(elements) (%v) != rows * columns (%v)\n", len(elements), rows*columns)
			return nil
		}

		for i := range elements {
			(*out)[i/columns][i%columns] = elements[i]
		}

		return out
	}
}

func Random(rows, columns int) *Matrix {
	r := Zeros(rows, columns)

	for i := range *r {
		for j := range (*r)[i] {
			(*r)[i][j] = rand.Float64()
		}
	}

	return r
}

func FromList(l []string) (*Matrix, error) {
	r := Zeros(len(l), 1)

	for i, s := range l {
		f, err := strconv.Atoi(s)

		if err != nil {
			return nil, err
		}

		(*r)[i][0] = float64(f)
	}

	return r, nil
}

func (m *Matrix) Add(n *Matrix) *Matrix {
	r := Zeros(len(*m), len((*m)[0]))

	for i := range *m {
		for j := range (*m)[i] {
			(*r)[i][j] = (*m)[i][j] + (*n)[i][j]
		}
	}

	return r
}

func (m *Matrix) AddScalar(c float64) *Matrix {
	r := Zeros(len(*m), len((*m)[0]))

	for i := range *m {
		for j := range (*m)[i] {
			(*r)[i][j] = (*m)[i][j] + c
		}
	}

	return r
}

func (m *Matrix) Subtract(n *Matrix) *Matrix {
	r := Zeros(len(*m), len((*m)[0]))

	for i := range *m {
		for j := range (*m)[i] {
			(*r)[i][j] = (*m)[i][j] - (*n)[i][j]
		}
	}

	return r
}

func (m *Matrix) SubtractScalar(c float64) *Matrix {
	r := Zeros(len(*m), len((*m)[0]))

	for i := range *m {
		for j := range (*m)[i] {
			(*r)[i][j] = (*m)[i][j] - c
		}
	}

	return r
}

func (m *Matrix) Multiply(n *Matrix) *Matrix {
	if len((*m)[0]) != len(*n) {
		fmt.Fprintf(os.Stderr, "can't muliply because rows (%v) != column (%v)\n", len(*m), len((*n)[0]))
		return nil
	}

	r := Zeros(len(*m), len((*n)[0]))

	for i := range *r {
		for k := range (*r)[0] {
			for j := range *n {
				(*r)[i][k] += (*m)[i][j] * (*n)[j][k]
			}
		}
	}

	return r
}

func (m *Matrix) MultiplyScalar(c float64) *Matrix {
	r := Zeros(len(*m), len((*m)[0]))

	for i := range *m {
		for j := range (*m)[i] {
			(*r)[i][j] = (*m)[i][j] * c
		}
	}

	return r
}

func (m *Matrix) MultiplyComponent(n *Matrix) *Matrix {
	r := Zeros(len(*m), len((*m)[0]))

	for i := range *r {
		for j := range (*r)[i] {
			(*r)[i][j] = (*m)[i][j] * (*n)[i][j]
		}
	}

	return r
}

func (m *Matrix) Transpose() *Matrix {
	r := Zeros(len((*m)[0]), len(*m))

	for i := range *m {
		for j := range (*m)[i] {
			(*r)[j][i] = (*m)[i][j]
		}
	}

	return r
}

func (m *Matrix) Apply(fn func(float64) float64) *Matrix {
	r := Zeros(len(*m), len((*m)[0]))

	for i := range *m {
		for j := range (*m)[i] {
			(*r)[i][j] = fn((*m)[i][j])
		}
	}

	return r
}

func (m *Matrix) Equal(n *Matrix) bool {
	if len(*m) != len(*n) {
		return false
	}

	for i := range *m {
		if len((*m)[i]) != len((*n)[i]) {
			return false
		}

		for j := range (*m)[i] {
			if (*m)[i][j] != (*n)[i][j] {
				return false
			}
		}
	}

	return true
}
