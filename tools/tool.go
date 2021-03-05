package tools

import (
	"encoding/csv"
	"errors"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func SumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func MakeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the dataset file.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 9

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices.
	inputsData := make([]float64, 6*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// Will track the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 6 || i == 7 || i == 8 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 6, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}
