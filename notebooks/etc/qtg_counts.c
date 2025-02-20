#include "knapsack.h"
#include "qtg_count.h"
#include "general_count.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Error: requires exactly 1 argument.\n");
        return 1;
    }
    char *path = calloc(strlen(argv[1]), sizeof(char));
    sprintf(path, "%s", argv[1]);

    knapsack_t* k = create_jooken_knapsack(path);
    if (k == NULL) {
        printf("Error: file not found\n");
        return 1;
    }
    count_t mixer = cycle_count_qtg_mixer(k, COPPERSMITH, TOFFOLI, 1);
    count_t state_prep = cycle_count_qtg(k, COPPERSMITH, TOFFOLI, 1);
    count_t phase = cycle_count_phase_separator();

    printf("%s,%d,%d,%d\n", k->name, state_prep, phase, mixer);


    free_knapsack(k);

    free(path);
}

