/*#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#define M_PI 3.14159265358979323846 
#define PRECISION 5

std::vector<double> worst_values;
double worst_fitness;
std::vector<double> best_values;
double best_fitness;
double stddev;
double final_pm;


struct chromosome {
	char* x;
	double fitness;
};

double calculate_standard_deviation(chromosome* population, int pop_size) {
	double sum = 0.0;
	for (int i = 0; i < pop_size; i++) {
		sum += 1 / population[i].fitness;
	}

	double mean = sum / pop_size;
	sum = 0.0;
	for (int i = 0; i < pop_size; i++) {
		sum += double(std::pow(1 / population[i].fitness - mean, 2) / pop_size);
	}
	double dev = std::sqrt(sum);
	dev = std::round(dev * 1e7) / 1e7;
	return dev;
}

void adjust_mutation_rate(double& pm, double std_dev, double std_dev_prev) {
	if (std_dev <= std_dev_prev)
		pm = std::min(pm + 0.0001, 0.01);
	else
		pm = std::max(pm - 0.0001, 0.0001);
}

void rand_x(chromosome c, int num_dims, int num_bits_per_dimension)
{
	int _length = num_dims * num_bits_per_dimension;
	for (int i = 0; i < _length; i++)
		c.x[i] = rand() % 2;
}
void copy_chromosome(chromosome* dest, chromosome source, int num_dims, int num_bits_per_dimension)
{
	int _length = num_dims * num_bits_per_dimension;
	for (int i = 0; i < _length; i++)
		dest->x[i] = source.x[i];
	dest->fitness = source.fitness;
}
double binary_to_real(char* b_string, unsigned long long num_bits_per_dimension, double min_x, double max_x)
{
	double x_real = 0;
	for (int j = 0; j < num_bits_per_dimension; j++)
		x_real = x_real * 2 + (int)b_string[j];
	x_real /= ((1ULL << num_bits_per_dimension) - 1);
	x_real *= (max_x - min_x);
	x_real += min_x;

	return x_real;
}
void compute_fitness_rastrigin(chromosome* c, int num_dims, int num_bits_per_dimension)
{
	double* x_real = new double[num_dims];
	for (int i = 0; i < num_dims; i++)
		x_real[i] = binary_to_real(c->x + i * num_bits_per_dimension, num_bits_per_dimension, -5.12, 5.12);
	c->fitness = 10 * num_dims;
	for (int i = 0; i < num_dims; i++)
		c->fitness += x_real[i] * x_real[i] - 10 * cos(2 * 3.1415 * x_real[i]);
	c->fitness = 1 / c->fitness;
	delete[] x_real;
}
void compute_fitness_griewangk(chromosome* c, int num_dims, int num_bits_per_dimension)
{
	double* x_real = new double[num_dims];
	for (int i = 0; i < num_dims; i++)
		x_real[i] = binary_to_real(c->x + i * num_bits_per_dimension, num_bits_per_dimension, -600, 600);
	c->fitness = 1;
	double sum_squares = 0.0;
	double product_cos = 1.0;
	for (int i = 0; i < num_dims; i++) {
		sum_squares += x_real[i] * x_real[i];
		product_cos *= cos(x_real[i] / sqrt(i + 1));
	}
	c->fitness += (sum_squares / 4000.0) - product_cos;
	c->fitness = 1 / c->fitness;
	delete[] x_real;
}
void compute_fitness_rosenbrock(chromosome* c, int num_dims, int num_bits_per_dimension)
{
	double* x_real = new double[num_dims];
	for (int i = 0; i < num_dims; i++)
		x_real[i] = binary_to_real(c->x + i * num_bits_per_dimension, num_bits_per_dimension, -2.048, 2.048);
	c->fitness = 0.0;
	for (int i = 0; i < num_dims - 1; i++) {
		c->fitness += 100 * pow(x_real[i + 1] - x_real[i] * x_real[i], 2) + pow(1 - x_real[i], 2);
	}
	c->fitness = 1 / c->fitness;
	delete[] x_real;
}
void compute_fitness_michalewicz(chromosome* c, int num_dims, int num_bits_per_dimension)
{
	double* x_real = new double[num_dims];
	for (int i = 0; i < num_dims; i++)
		x_real[i] = binary_to_real(c->x + i * num_bits_per_dimension, num_bits_per_dimension, 0, M_PI);
	c->fitness = 0.0;
	for (int i = 0; i < num_dims; i++) {
		c->fitness -= sin(x_real[i]) * pow(sin((i + 1) * x_real[i] * x_real[i] / M_PI), 20);
	}
	c->fitness += 100;
	c->fitness = 1 / c->fitness;
	delete[] x_real;
}
void compute_fitness(chromosome* c, int num_dims, int num_bits_per_dimension, int function_option)
{
	switch (function_option) {
	case 1:
		compute_fitness_rastrigin(c, num_dims, num_bits_per_dimension);
		break;
	case 2:
		compute_fitness_griewangk(c, num_dims, num_bits_per_dimension);
		break;
	case 3:
		compute_fitness_rosenbrock(c, num_dims, num_bits_per_dimension);
		break;
	case 4:
		compute_fitness_michalewicz(c, num_dims, num_bits_per_dimension);
		break;
	}
}
void mutation(chromosome* c, int num_dims, int num_bits_per_dimension, double pm)
{
	int _length = num_dims * num_bits_per_dimension;
	double p;
	for (int i = 0; i < _length; i++) {
		p = rand() / (double)RAND_MAX;
		if (p < pm)
			c->x[i] = 1 - c->x[i];
	}
}
void three_cut_point_crossover(chromosome parent1, chromosome parent2, chromosome* offspring1, chromosome* offspring2, int num_dims, int num_bits_per_dimension)
{
	int pct1, pct2, pct3;
	int length = num_dims * num_bits_per_dimension;
	pct1 = 1 + rand() % (length - 2);
	pct2 = 1 + rand() % (length - 2);
	pct3 = 1 + rand() % (length - 2);
	if (pct1 > pct2) {
		std::swap(pct1, pct2);
	}
	if (pct2 > pct3) {
		std::swap(pct2, pct3);
	}
	if (pct1 > pct3) {
		std::swap(pct1, pct3);
	}
	for (int i = 0; i < pct1; i++) {
		offspring1->x[i] = parent1.x[i];
		offspring2->x[i] = parent2.x[i];
	}
	for (int i = pct1; i < pct2; i++) {
		offspring1->x[i] = parent2.x[i];
		offspring2->x[i] = parent1.x[i];
	}
	for (int i = pct2; i < pct3; i++) {
		offspring1->x[i] = parent1.x[i];
		offspring2->x[i] = parent2.x[i];
	}
	for (int i = pct3; i < num_dims * num_bits_per_dimension; i++) {
		offspring1->x[i] = parent2.x[i];
		offspring2->x[i] = parent1.x[i];
	}
}
int sort_function(const void* a, const void* b)
{
	if (((chromosome*)a)->fitness > ((chromosome*)b)->fitness)
		return -1;
	else
		if (((chromosome*)a)->fitness < ((chromosome*)b)->fitness)
			return 1;
		else
			return 0;
}
void get_minmax(int function_option, double* min_x, double* max_x) {
	switch (function_option) {
	case 1:
		*min_x = -5.12;
		*max_x = 5.12;
		break;
	case 2:
		*min_x = -600;
		*max_x = 600;
		break;
	case 3:
		*min_x = -2.048;
		*max_x = 2.048;
		break;
	case 4:
		*min_x = 0;
		*max_x = M_PI;
		break;
	}
}
void print_chromosome(chromosome* population, int pop_size, int num_dims, int num_bits_per_dimension, int function_option, double std_dev, double pm)
{
	double min_x, max_x;
	get_minmax(function_option, &min_x, &max_x);

	chromosome* best_chromosome = &population[0];
	chromosome* worst_chromosome = &population[pop_size - 1];

	printf("Best: x = (");
	best_values.clear();
	for (int i = 0; i < num_dims; i++) {
		double x_real = binary_to_real(best_chromosome->x + i * num_bits_per_dimension, num_bits_per_dimension, min_x, max_x);
		best_values.push_back(x_real);
		printf("%lf ", x_real);
	}
	printf(") ");
	printf("fitness = %lf ", best_chromosome->fitness);
	if (function_option == 4) {
		printf("f(x) = %lf", (1 / best_chromosome->fitness) - 100);
		best_fitness = (1 / best_chromosome->fitness) - 100;
	}
	else {
		printf("f(x) = %lf", 1 / best_chromosome->fitness);
		best_fitness = 1 / best_chromosome->fitness;
	}

	printf("\n");

	printf("Worst: x = (");
	worst_values.clear();
	for (int i = 0; i < num_dims; i++) {
		double x_real = binary_to_real(worst_chromosome->x + i * num_bits_per_dimension, num_bits_per_dimension, min_x, max_x);
		worst_values.push_back(x_real);
		printf("%lf ", x_real);
	}
	printf(") ");
	if (function_option == 4) {
		printf("f(x) = %lf", (1 / worst_chromosome->fitness) - 100);
		worst_fitness = (1 / worst_chromosome->fitness) - 100;
	}
	else {
		printf("f(x) = %lf", 1 / worst_chromosome->fitness);
		worst_fitness = 1 / worst_chromosome->fitness;
	}
	printf("\n");

	stddev = std_dev;
	final_pm = pm;

	printf("Standard deviation = %f, Mutation probability = %f\n\n", std_dev, pm);
}
void tournament_selection(int* r1, int* r2, int tournament_size, int pop_size)
{
	int i;
	*r1 = pop_size;
	*r2 = pop_size;
	for (int j = 0; j < tournament_size; j++) {
		i = rand() % pop_size;
		if (i < *r1) {
			*r2 = *r1;
			*r1 = i;
		}
		else if (i < *r2)
			*r2 = i;
	}
}

void hill_climbing(chromosome* dest, int num_dims, int num_bits_per_dimension, int function_option, int steps)
{
	int t = 0, new_steps = 0;
	chromosome neighbour;
	neighbour.x = (char*)malloc(num_dims * num_bits_per_dimension);
	bool local = false;
	int _length = num_dims * num_bits_per_dimension;
	while (!local && steps > 0) {
		local = true;
		copy_chromosome(&neighbour, *dest, num_dims, num_bits_per_dimension);
		for (int i = 0; i < _length; i++) {
			neighbour.x[i] = 1 - neighbour.x[i];
			compute_fitness(&neighbour, num_dims, num_bits_per_dimension, function_option);
			if (neighbour.fitness > dest->fitness) {
				dest->x[i] = 1 - dest->x[i];
				dest->fitness = neighbour.fitness;
				local = false;
				new_steps = steps - 1;
			}
			neighbour.x[i] = 1 - neighbour.x[i];
		}

	}
}

void wheel_of_fortune_selection(chromosome* population, chromosome* new_population, int pop_size, int num_dims, int num_bits_per_dimension) {
	double* eval = (double*)malloc(pop_size * sizeof(double));
	double total_fitness = 0.0;
	double* selection_prob = (double*)malloc(pop_size * sizeof(double));
	double* accumulated_prob = (double*)malloc((pop_size + 1) * sizeof(double));

	for (int i = 0; i < pop_size; i++) {
		eval[i] = population[i].fitness;
		total_fitness += eval[i];
	}

	for (int i = 0; i < pop_size; i++) {
		selection_prob[i] = eval[i] / total_fitness;
	}

	accumulated_prob[0] = 0.0;
	for (int i = 0; i < pop_size; i++) {
		accumulated_prob[i + 1] = accumulated_prob[i] + selection_prob[i];
	}

	for (int i = 0; i < pop_size; i++) {
		double r = (double)rand() / RAND_MAX;

		for (int j = 0; j < pop_size; j++) {
			if (accumulated_prob[j] < r && r <= accumulated_prob[j + 1]) {
				copy_chromosome(&new_population[i], population[j], num_dims, num_bits_per_dimension);
				break;
			}
		}
	}
}


int calculate_num_bits_per_dimension(int function_option, double precision) {
	double min_x, max_x;
	get_minmax(function_option, &min_x, &max_x);
	return static_cast<int>(ceil(log2((max_x - min_x) * pow(10, precision))));
}

void start_steady_state_ga(int pop_size, int num_gens, int num_dims, int num_bits_per_dimension, double pcross, double pm, int function_option, int hc_option, int tournament_size)
{
	chromosome* population;
	int k;
	population = (chromosome*)malloc(pop_size * sizeof(chromosome));
	chromosome* new_population;
	new_population = (chromosome*)malloc(pop_size * sizeof(chromosome));
	for (int i = 0; i < pop_size; i++) {
		population[i].x = (char*)malloc(num_dims * num_bits_per_dimension);
		new_population[i].x = (char*)malloc(num_dims * num_bits_per_dimension);
	}
	chromosome offspring1, offspring2;
	offspring1.x = (char*)malloc(num_dims * num_bits_per_dimension);
	offspring2.x = (char*)malloc(num_dims * num_bits_per_dimension);
	for (int i = 0; i < pop_size; i++) {
		rand_x(population[i], num_dims, num_bits_per_dimension);
		compute_fitness(&population[i], num_dims, num_bits_per_dimension, function_option);
	}
	qsort((void*)population, pop_size, sizeof(population[0]), sort_function);
	printf("generation 0\n");
	double std_dev_prev = calculate_standard_deviation(population, pop_size);
	print_chromosome(population, pop_size, num_dims, num_bits_per_dimension, function_option, std_dev_prev, pm);
	std::vector<int> indices(pop_size);
	for (int i = 0; i < pop_size; i++) indices[i] = i;
	std::random_shuffle(indices.begin(), indices.end());
	for (int g = 1; g <= num_gens; g++) {
		if (hc_option / 2 == 1) {
			for (int i = 0; i < pop_size; i++)
				hill_climbing(&population[i], num_dims, num_bits_per_dimension, function_option, g*10);
		}
		for (int k = 0; k < pop_size / 20; k++) {
			copy_chromosome(&new_population[k], population[k], num_dims, num_bits_per_dimension);
		}
		for (int k = pop_size / 20; k < pop_size; k++) {
			int r1, r2;
			tournament_selection(&r1, &r2, tournament_size, pop_size);
			copy_chromosome(&new_population[k], population[r1], num_dims, num_bits_per_dimension);
		}
		
		for (int k = 0; k < pop_size - 1; k += 2) {
			int r1 = indices[k];
			int r2 = indices[k + 1];
			double p = rand() / double(RAND_MAX);
			if (p < pcross)
				three_cut_point_crossover(new_population[r1], new_population[r2], &offspring1, &offspring2, num_dims, num_bits_per_dimension);
			else {
				copy_chromosome(&offspring1, new_population[r1], num_dims, num_bits_per_dimension);
				copy_chromosome(&offspring2, new_population[r2], num_dims, num_bits_per_dimension);
			}
			mutation(&offspring1, num_dims, num_bits_per_dimension, pm);
			compute_fitness(&offspring1, num_dims, num_bits_per_dimension, function_option);
			mutation(&offspring2, num_dims, num_bits_per_dimension, pm);
			compute_fitness(&offspring2, num_dims, num_bits_per_dimension, function_option);
			if (hc_option % 2 == 1) {
				hill_climbing(&offspring1, num_dims, num_bits_per_dimension, function_option, g*10);
				hill_climbing(&offspring2, num_dims, num_bits_per_dimension, function_option, g*10);
			}
			copy_chromosome(&new_population[k], offspring1, num_dims, num_bits_per_dimension);
			copy_chromosome(&new_population[k + 1], offspring2, num_dims, num_bits_per_dimension);
		}
		qsort((void*)new_population, pop_size, sizeof(new_population[0]), sort_function);
		for (int k = 0; k < pop_size / 20; k++) {
			if (new_population[k].fitness > population[k].fitness)
				copy_chromosome(&population[k], new_population[k], num_dims, num_bits_per_dimension);
		}
		for (int k = pop_size / 20; k < pop_size; k++) {
			copy_chromosome(&population[k], new_population[k], num_dims, num_bits_per_dimension);
		}
		double std_dev = calculate_standard_deviation(population, pop_size);
		adjust_mutation_rate(pm, std_dev, std_dev_prev);
		std_dev_prev = std_dev;
		printf("generation %d\n", g);
		print_chromosome(population, pop_size, num_dims, num_bits_per_dimension, function_option, std_dev, pm);
	}


	free(offspring1.x);
	free(offspring2.x);
	for (int i = 0; i < pop_size; i++)
		free(population[i].x);
	free(population);
	free(new_population);
}


template <typename T>
std::string to_string(const T& value) {
	std::ostringstream oss;
	oss << value;
	return oss.str();
}



std::string vector_to_string(const std::vector<double>& values) {
	std::ostringstream oss;
	for (size_t i = 0; i < values.size(); ++i) {
		oss << std::fixed << std::setprecision(PRECISION) << values[i];
		if (i < values.size() - 1) {
			oss << "; ";
		}
	}
	return oss.str();
}

int main(void)
{
	int pop_size;
	int num_gens;
	double pm = 0.001;
	double pcross = 0.8;
	int num_dims;
	std::cout << "Enter number of dimensions(2,30,100,others): ";
	std::cin >> num_dims;
	//sugerate la ora, dar toate fct cu care lucram au min si max in definitie deci ????
	//double min_x = -1e10;
	//double max_x = 1e10;
	int function_option = 1;
	int hc_option = 1;
	int tournament_size = 5;
	std::cout << "Enter function option (1-4): ";
	std::cin >> function_option;
	std::cout << "Enter hill climbing option (1: hc children, 2: hc all at start of generation, 3: both, 4: neither): ";
	std::cin >> hc_option;

	if (hc_option == 4) {
		pop_size = 10000;
		num_gens = 1000;
	}
	else {
		pop_size = 1000;
		num_gens = 100;
	}

	int num_bits_per_dimension = calculate_num_bits_per_dimension(function_option, PRECISION); //max is for fct3: needs at least 27. i rouded up
	printf("Number of bits per dimension %d \n", num_bits_per_dimension);
	srand(time(0));

	clock_t start_time = clock();
	start_steady_state_ga(pop_size, num_gens, num_dims, num_bits_per_dimension, pcross, pm, function_option, hc_option, tournament_size);
	clock_t end_time = clock();
	double time_measurement = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

	printf("Press enter ...");
	getchar();

	std::ostringstream filename;
	filename << "results_function_" << function_option << "_hc_" << hc_option << "_num_dims_" << num_dims << ".csv";
	std::string filename_str = filename.str();

	bool file_exists = false;
	std::ifstream file_check(filename_str);
	if (file_check.good()) {
		file_exists = true;
	}
	file_check.close();


	std::ofstream file(filename_str, std::ios::app);
	if (file.is_open()) {
		if (!file_exists) {
			file << "function_option,hc_option,pop_size,num_gens,pcross,pm,num_dims,"
				<< "tournament_size,num_bits_per_dimension,time_measurement,"
				<< "best_fitness,best_x,worst_fitness,worst_x,std_deviation,final_pm\n";
		}

		file << function_option << "," << hc_option << "," << pop_size << "," << num_gens << ","
			<< pcross << "," << pm << "," << num_dims << "," << tournament_size << ","
			<< num_bits_per_dimension << "," << time_measurement << ","
			<< best_fitness << ",";

		file << "[" << vector_to_string(best_values) << "]," << worst_fitness << ",";

		file << "[" << vector_to_string(worst_values) << "]," << stddev << ", " << final_pm << "\n";

		file.close();
		std::cout << "Results and hyperparameters saved to '" << filename_str << "'.\n";
	}
	else {
		std::cerr << "Unable to open file for writing.\n";
	}

	return 0;
}
*/