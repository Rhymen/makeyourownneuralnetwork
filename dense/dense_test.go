package dense

import "testing"

func TestZeros(t *testing.T) {
	m := Zeros(2, 1)

	n := New(2, 1)(
		0,
		0,
	)

	if !m.Equal(n) {
		t.FailNow()
	}
}

func TestNew(t *testing.T) {
	m := New(3, 3)(
		0, 1, 2,
		1, 2, 3,
		2, 3, 4,
	)

	for i := range m {
		for j := range m[i] {
			if int(m[i][j]) != (i + j) {
				t.FailNow()
			}
		}
	}
}

func TestRandom(t *testing.T) {
	m := Random(3, 3)

	n := Random(3, 3)

	z := Zeros(3, 3)

	if m.Equal(n) || m.Equal(z) {
		t.FailNow()
	}
}

func TestFromList(t *testing.T) {
	m := New(3, 1)(
		1,
		2,
		3,
	)

	l := []string{"1", "2", "3"}

	n, err := FromList(l)
	if err != nil || !m.Equal(n) {
		t.FailNow()
	}
}

func TestMatrix_Add(t *testing.T) {
	m := New(3, 3)(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	)

	n := New(3, 3)(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	)

	rr := New(3, 3)(
		0, 2, 4,
		6, 8, 10,
		12, 14, 16,
	)

	r := m.Add(n)

	if !r.Equal(rr) {
		t.FailNow()
	}
}

func TestMatrix_AddScalar(t *testing.T) {
	m := New(3, 3)(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	)

	c := 10.0

	rr := New(3, 3)(
		10, 11, 12,
		13, 14, 15,
		16, 17, 18,
	)

	r := m.AddScalar(c)

	if !r.Equal(rr) {
		t.FailNow()
	}
}

func TestMatrix_Subtract(t *testing.T) {
	m := New(3, 3)(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	)

	n := New(3, 3)(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	)

	rr := Zeros(3, 3)

	r := m.Subtract(n)

	if !r.Equal(rr) {
		t.FailNow()
	}
}

func TestMatrix_SubtractScalar(t *testing.T) {
	m := New(3, 3)(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	)

	c := 10.0

	rr := New(3, 3)(
		-10, -9, -8,
		-7, -6, -5,
		-4, -3, -2,
	)

	r := m.SubtractScalar(c)

	if !r.Equal(rr) {
		t.FailNow()
	}
}

func TestMatrix_Multiply(t *testing.T) {
	m := New(4, 3)(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11,
	)

	n := New(3, 1)(
		1,
		3,
		6,
	)

	rr := New(4, 1)(
		15,
		45,
		75,
		105,
	)

	r := m.Multiply(n)

	if !r.Equal(rr) {
		t.FailNow()
	}
}

func TestMatrix_MultiplyScalar(t *testing.T) {
	m := New(3, 3)(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	)

	c := 10.0

	rr := New(3, 3)(
		0, 10, 20,
		30, 40, 50,
		60, 70, 80,
	)

	r := m.MultiplyScalar(c)

	if !r.Equal(rr) {
		t.FailNow()
	}
}

func TestMatrix_MultiplyComponent(t *testing.T) {
	m := New(3, 1)(
		1,
		2,
		3,
	)

	n := New(3, 1)(
		1,
		2,
		3,
	)

	rr := New(3, 1)(
		1,
		4,
		9,
	)

	r := m.MultiplyComponent(n)

	if !r.Equal(rr) {
		t.FailNow()
	}
}

func Test_Transpose(t *testing.T) {
	m := New(2, 4)(
		1, 2, 3, 4,
		5, 6, 7, 8,
	)

	rr := New(4, 2)(
		1, 5,
		2, 6,
		3, 7,
		4, 8,
	)

	r := m.Transpose()

	if !r.Equal(rr) {
		t.FailNow()
	}
}
