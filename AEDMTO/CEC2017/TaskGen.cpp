#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#define tasknum 9
// A General method to generate CEC2017 task code for python version
int main(int argc, char const *argv[])
{
    // string Popclass;
    // cin >> Popclass;
    ofstream outfile("task.txt", ios::out);
    string tasks[tasknum] = {"CI_HS", "CI_MS", "CI_LS", "PI_HS", "PI_MS", "PI_LS", "NI_HS", "NI_MS", "NI_LS"};
    outfile << "Ps = [" << endl;
    // for (int i = 0; i < tasknum; i++)
    // {
    //     outfile << "\t\t" << Popclass << "(" << tasks[i] << "())," << endl;
    // }
    for (int i = 0; i < tasknum; i++)
    {
        outfile << "\t\t" << tasks[i] <<"(), " << endl;
    }
    outfile << "]" << endl;
    outfile << "It's a general method to generate CEC2017 task code for python version." << endl;
    outfile << "Copy the codes above to the python code file." << endl;
    return 0;
}
