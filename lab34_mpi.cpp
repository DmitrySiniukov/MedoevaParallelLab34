#include "mpi.h"
#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <numeric>
using namespace std;

const vector<double> DEFAULT_COEFF_B(7, 1.0);
const double PI = 3.14159265359;

vector<double> findVectorOfCoefB(const vector<int> &, const vector<double> &, const vector<double> &);
vector<vector<double> > transpose(const vector<vector<double> > &);
vector<vector<double> > operator*(const vector<vector<double> >& a, const vector<vector<double> >& b);
vector<double> calculate(const vector<int>&, double, const vector<double>& = DEFAULT_COEFF_B);
double CalcDeterminant(vector<vector<double> >, int);
vector<vector<double> > GetMinor(vector<vector<double> >, int, int, int);
vector<vector<double> > inverse(vector<vector<double> >);
string binary(unsigned);
double regularityComparation(const vector<int>&, const vector<double>&, const vector<double>&, const vector<double>&);
double biasComparation(const vector<int>&, const vector<double>&, const vector<double>&, const vector<double>&, const vector<double>&);

int main(int argc, char * argv[]) {
	int numtasks, taskid;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

	vector<double> yA;
	vector<double> xA;
	vector<double> yB;
	vector<double> xB;
	vector<double> yTable;
	vector<double> xTable;

	ifstream ifile("tempByDays.txt");

	if (!ifile.is_open()) {
		cerr << "There was a problem opening the input file!\n";
		exit(0);
	}

	double t = 0.0;
	int indxCounter = 0;
	while (ifile >> t) {
		indxCounter++;
		if (yA.size() < yB.size()) {
			yA.push_back(t);
			xA.push_back(indxCounter);
		}
		else {
			yB.push_back(t);
			xB.push_back(indxCounter);
		}
		xTable.push_back(indxCounter);
		yTable.push_back(t);
	}

	vector <vector<int> > L;
	for (int i = 1; i < 64; ++i)
	{
		L.push_back(vector<int>(1, 1));
		string binary_number = binary(i);
		for (int j = 0; j < binary_number.size(); ++j) {
			if (binary_number.compare(j, 1, "1") == 0) {
				L[i - 1].push_back(j + 2);
			}
		}
	}

	if (taskid != 0) {
		float start_time = MPI_Wtime();
		int nModel = L.size() / (numtasks - 1);
		int start = (taskid - 1) * nModel;
		int end = taskid * nModel;

		vector<vector<double> > bA;
		for (int i = start; i < end; i++)
			bA.push_back(vector<double>(findVectorOfCoefB(L[i], xA, yA)));

		vector<vector<double> > bB;
		for (int i = start; i < end; i++)
			bB.push_back(vector<double>(findVectorOfCoefB(L[i], xB, yB)));

		for (int i = 0; i < bA.size(); i++) {
			MPI_Send(&bA[i][0], L[(taskid - 1) * nModel + i].size(), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			MPI_Send(&bB[i][0], L[(taskid - 1) * nModel + i].size(), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
		}
	}
	if (taskid == 0) {
		int nModel = L.size() / (numtasks - 1);
		vector<vector<double> > results_bA;
		vector<vector<double> > results_bB;

		for (int p = 1; p < numtasks; p++) {
			for (int i = ((p - 1) * nModel); i < p * nModel; i++) {
				vector<double> tempA(L[i].size());
				vector<double> tempB(L[i].size());

				MPI_Recv(&tempA[0], L[i].size(), MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				results_bA.push_back(tempA);

				MPI_Recv(&tempB[0], L[i].size(), MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				results_bB.push_back(tempB);
			}
		}

		int nModelTail = L.size() - ((L.size() / (numtasks - 1)) * (numtasks - 1));
		// cout << "MASTER nModel: " << nModelTail << endl;
		if (nModelTail > 0) {
			int start = L.size() - nModelTail;
			int end = L.size();

			vector<vector<double> > bA;
			for (int i = start; i < end; i++)
				results_bA.push_back(vector<double>(findVectorOfCoefB(L[i], xA, yA)));

			vector<vector<double> > bB;
			for (int i = start; i < end; i++)
				results_bB.push_back(vector<double>(findVectorOfCoefB(L[i], xB, yB)));
		}

		/*Final comparation*/
		vector<double> square_comparations;
		for (int i = 0; i < results_bA.size(); ++i)
			square_comparations.push_back(regularityComparation(L[i], results_bA[i], xB, yB));


		int min_pos = distance(square_comparations.begin(), min_element(square_comparations.begin(), square_comparations.end()));
		cout << "The best model by SQR criteria is the model #" << min_pos + 1 << " (" << square_comparations[min_pos] << ") " << " with b: ";
		for (int j = 0; j < results_bA[min_pos].size(); ++j)
			cout << results_bA[min_pos][j] << " ";

		cout << "// with model functions #: ";
		for (int j = 0; j < L[min_pos].size(); ++j)
			cout << L[min_pos][j] << " ";

		vector<double> bias_comparation;
		if (results_bA.size() != results_bB.size()) {
			cout << "bA and bB must be the same length";
			exit(0);
		}

		for (int i = 0; i < results_bA.size(); ++i)
			bias_comparation.push_back(biasComparation(L[i], results_bA[i], results_bB[i], xTable, yTable));

		min_pos = distance(bias_comparation.begin(), min_element(bias_comparation.begin(), bias_comparation.end()));
		cout << "\nThe best model by DIFFERENCE criteria is the model #" << min_pos + 1 << " (" << bias_comparation[min_pos] << ") " << " with b: ";
		for (int j = 0; j < results_bA[min_pos].size(); ++j)
			cout << results_bA[min_pos][j] << " ";

		cout << "// with model functions #: ";
		for (int j = 0; j < L[min_pos].size(); ++j)
			cout << L[min_pos][j] << " ";
		cout << endl;
		float end_time = MPI_Wtime();
        printf("\nRunning Time = %f\n\n", end_time - start_time);
	}

	MPI_Finalize();
}

// Find coefficients of b for the model
// L - indexes of used functions
// XA - 1d vector of x from A
// YA - 1d vector of y from A
vector<double> findVectorOfCoefB(const vector<int> &L, const vector<double> &XA, const vector<double> &YA) {
	vector<vector<double> >     X_functions,    // matrix of the model's functions results
		X_transposed,
		YT(1, YA),
		Y;

	for (int i = 0; i < XA.size(); ++i)
		X_functions.push_back(calculate(L, XA[i]));

	X_transposed = transpose(X_functions);

	Y = transpose(YT);
	vector<double> calculated_bs(transpose((inverse((X_transposed*X_functions))*X_transposed)*Y)[0]);

	return calculated_bs;
}

// Transpose a 2d matrix
vector<vector<double> > transpose(const vector<vector<double> > &matrix) {
	int size[] = { (int)matrix.size(), (int)matrix[0].size() };
	vector<vector<double> > transposedMatrix(size[1], vector<double>(size[0], 0));

	for (int i = 0; i < matrix.size(); ++i)
		for (int j = 0; j < matrix[i].size(); ++j)
			transposedMatrix[j][i] = matrix[i][j];

	return transposedMatrix;
};


vector<double> calculate(const vector<int>& indxFunc, double x, const vector<double>& b) {
	vector<double> calcVal;

	for (int i = 0; i < indxFunc.size(); i++) {
		switch (indxFunc[i]) {
		case 1:
			calcVal.push_back(b[0] * 1);
			break;

		case 2:
			calcVal.push_back(b[1] * sin(2 * PI * x / 365));
			break;

		case 3:
			calcVal.push_back(b[2] * cos(2 * PI * x / 365));
			break;

		case 4:
			calcVal.push_back(b[3] * sin(24 * PI * x / 365));
			break;

		case 5:
			calcVal.push_back(b[4] * cos(24 * PI * x / 365));
			break;

		case 6:
			calcVal.push_back(b[5] * sin(PI * x / 14));
			break;

		case 7:
			calcVal.push_back(b[6] * cos(PI * x / 14));
			break;

		default:
			cout << "Something wrong" << endl;
			break;
		}
	}

	return calcVal;
}

vector<vector<double> > operator*(const vector<vector<double> >& a, const vector<vector<double> >& b) {
	vector<vector<double> > result(a.size(), vector<double>(b[0].size()));
	for (size_t i = 0; i < a.size(); i++)
	{
		for (size_t j = 0; j < b[0].size(); j++)
		{
			double sum = 0;
			for (size_t k = 0; k < a[0].size(); k++)
			{
				sum += a[i][k] * b[k][j];
			}
			result[i][j] = sum;
		}
	}

	return  result;
}

// matrix inversioon
vector<vector<double> > inverse(vector<vector<double> > matrix) {
	int order = matrix.size();
	vector<vector<double> > res(order, vector<double>(order));
	// get the determinant of matrix
	double det = 1.0 / CalcDeterminant(matrix, order);

	// memory allocation
	vector<vector<double> > minorM(order - 1, vector<double>(order - 1));


	for (int j = 0; j < order; j++)
	{
		for (int i = 0; i < order; i++)
		{
			// get the co-factor (matrix) of A(j,i)
			minorM = GetMinor(matrix, j, i, order);
			res[i][j] = det*CalcDeterminant(minorM, order - 1);
			if ((i + j) % 2 == 1)
				res[i][j] = -res[i][j];
		}
	}

	return res;
}

// calculate the cofactor of element (row,col)
vector<vector<double> > GetMinor(vector<vector<double> > src, int row, int col, int order) {
	// indicate which col and row is being copied to dest
	int colCount = 0, rowCount = 0;
	vector<vector<double> > res(order, vector<double>(order));

	for (int i = 0; i < order; i++)
	{
		if (i != row)
		{
			colCount = 0;
			for (int j = 0; j < order; j++)
			{
				// when j is not the element
				if (j != col)
				{
					res[rowCount][colCount] = src[i][j];
					colCount++;
				}
			}
			rowCount++;
		}
	}

	return res;
}

// Calculate the determinant recursively.
double CalcDeterminant(vector<vector<double> > matrix, int order) {
	//order must be >= 0
	//stop the recursion when matrix is a single element
	if (order == 1)
		return matrix[0][0];

	// the determinant value
	double det = 0;

	// allocate the cofactor matrix
	vector<vector<double> > minorM(order - 1, vector<double>(order - 1));

	for (int i = 0; i < order; i++)
	{
		// get minor of element (0,i)
		minorM = GetMinor(matrix, 0, i, order);
		// the recusion is here!

		det += (i % 2 == 1 ? -1.0 : 1.0) * matrix[0][i] * CalcDeterminant(minorM, order - 1);
	}

	return det;
}

// Convert int to binary string
string binary(unsigned x) {
	string s;
	do {
		s.push_back('0' + (x & 1));
	} while (x >>= 1);

	reverse(s.begin(), s.end());
	for (int i = s.size(); i < 6; i++)
		s.insert(s.begin(), '0');

	return s;
}

//regularity criterion
double regularityComparation(const vector<int> &L, const vector<double> &b, const vector<double> &xB, const vector<double> &yB) {
	double  model_minus_table_sqr_sum = 0.0,
		table_sqr_sum = 0.0;

	for (int i = 0; i < xB.size(); ++i)
	{
		vector<double> model_res_vector = calculate(L, xB[i], b);
		double  model_res = accumulate(model_res_vector.begin(), model_res_vector.end(), 0.0);
		double  table_res = yB[i];

		model_minus_table_sqr_sum += pow(model_res - table_res, 2);
		table_sqr_sum += pow(table_res, 2);
	}

	return model_minus_table_sqr_sum / table_sqr_sum;
}

//the criterion of minimum shift
double biasComparation(const vector<int> &L, const vector<double> &bA, const vector<double> &bB, const vector<double> &xTable, const vector<double> &yTable) {
	double  models_diff_sqr_sum = 0.0,
		table_sqr_sum = 0.0;

	for (int i = 0; i < xTable.size(); ++i)
	{
		vector<double> model_res_vector_A(calculate(L, xTable[i], bA));
		vector<double> model_res_vector_B(calculate(L, xTable[i], bB));

		double  model_res_A = accumulate(model_res_vector_A.begin(), model_res_vector_A.end(), 0.0);
		double  model_res_B = accumulate(model_res_vector_B.begin(), model_res_vector_B.end(), 0.0);
		double  table_res = yTable[i];

		models_diff_sqr_sum += pow(model_res_A - model_res_B, 2);
		table_sqr_sum += pow(table_res, 2);
	}

	return models_diff_sqr_sum / table_sqr_sum;
}