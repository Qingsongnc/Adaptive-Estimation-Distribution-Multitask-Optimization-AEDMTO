#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#define tasknum 10
// A General method to generate CEC2017 task code for python version
int main(int argc, char const *argv[])
{
    // string Popclass;
    // cin >> Popclass;
    ofstream outfile("task.txt", ios::out);
    outfile << "Ps = [" << endl;
    for (int i = 0; i < tasknum; i++)
    {
        outfile << "\t\t" << "Benchmark" << i + 1 << "()," << endl;
    }
    outfile << "]" << endl;
    outfile << "It's a general method to generate CEC2022 task code for python version." << endl;
    outfile << "Copy the codes above to the python code file." << endl;
    return 0;
}
