#include <stdlib.h>
#include <stdio.h>
#define MIN_DOUBLE -1.79769e+308

typedef struct linked_list{
    struct linked_list* next;
    struct linked_list* prev;
    int value;
} linked_list;



void print_array_stats(double* array, int size){
    double sum = 0;
    double largest = MIN_DOUBLE;
    double average = 0;
    for (int i = 0; i<size; i++) {
        sum += array[i];
        if (array[i] > largest) largest = array[i];
    }
    average = sum/(double)size;
    printf("Array sum = %f\n", sum);
    printf("Array largest = %f\n", largest);
    printf("Array average = %f\n", average);
}


linked_list* new_linked_list(int size, int value){
    linked_list* list = (linked_list*)malloc(sizeof(linked_list)*size);
    //can treat linked list as array because elements are sequential in memory
    for (int i = 0; i<size; i++) {
        list[i].value = value;
        if (i == 0) { //if looking at first element, previous element is null
            list[i].next = &list[i + 1];
        }
        else if (i == size - 1) { //if looking at last element, next element is null
            list[i].prev = &list[i - 1];
        }
        else{ //normal case
            list[i].prev = &list[i - 1];
            list[i].next = &list[i + 1];
        }       
    }
    return list;
}


void print_linked_list(linked_list* ll, int horizontal, int direction){
    linked_list* curr = ll;
    int* values = (int*)malloc(sizeof(int)*100);
    int i = 0;
    //Iterate through linked list, and save values in an array. If direction is true the variables are printed. If direction is false, the variables are saved in the first iteration and then the array is iterated through backwards and printed.  
    while (curr != NULL) { 
        values[i] = (*curr).value;      
        if (direction) {  
            if (!horizontal) printf("%i\n", values[i]); //prints values with line change if !horizontal
            else printf("%i", values[i]);   
        }
            curr = (*curr).next;
            i++;
    }
    if (!direction) {
        for (int j = i - 1; j > -1; j--) {
            if (!horizontal) printf("%i\n ", values[i]); //prints values with line change if !horizontal
            else printf("%i", values[i]);
        }
    }
}

int sum_linked_list(linked_list* ll){
    int sum = 0;
    linked_list* curr = ll; 
    while (curr != NULL) {
        sum += curr->value;
        curr = curr->next;
    }
    return sum;
}

void insert_linked_list(linked_list* ll, int pos, int value){
    //create a new linked_list element on the heap
    linked_list* element = (linked_list*)malloc(sizeof(linked_list)); 
    //insert this element in the linked list 
    element->value = value;
    element->next = &ll[pos];
    element->prev = &ll[pos - 1];
    //update prev and next for the elements before and after the inserted element
    ll[pos].prev= element;
    ll[pos - 1].next = element;
}


void merge_linked_list(linked_list* a, linked_list* b){
    a->next = b;

    linked_list* curr = b;   
    linked_list* last_from_a = a; //variable that holds pointer to the last element you looked at from a
    linked_list* last_from_b = b; //variable that holds pointer to the last element you looked at from b

    int i = 0; 

    while (curr != NULL) {
        if (i % 2 == 0) { //even number -> take element from b 
            curr->next = (*last_from_a).next;
            curr->prev = last_from_a;
            last_from_b = curr;
        }
        else { // odd number -> take element from a
            curr->next = (*last_from_b).next;
            curr->prev = last_from_b;
            last_from_a = curr;
        }
        curr = curr->next;
        i++;
    }
}

void destroy_linked_list(linked_list* ll){
    linked_list* curr = ll;
    while (curr != NULL) {
        linked_list* next = curr->next; 
        free(curr); 
        curr = next; 
    }
}  

int main(int argc, char** argv){

    //Array statistics
    //double array[5] = {2.0, 3.89, -3.94, 10.1, 0.88};
    //print_array_stats(array, 5);


    //Creating liked list with 3 3s and 4 4s
    linked_list* ll3 = new_linked_list(3,3);
    linked_list* ll4 = new_linked_list(4,4);

    //Should print: "3 3 3"
    //print_linked_list(ll3, 1, 1);

    //Inserting a 5 at the 1st position
    insert_linked_list(ll3, 1, 5);


    //Should print "3 5 3 3"
    //print_linked_list(ll3, 1, 1);

    //Printing backwards, should print: "3 3 5 3"
    //print_linked_list(ll3, 1, 0);

    //Merging the linked lists
    merge_linked_list(ll3, ll4);

    //Printing the result, should print: "3 4 5 4 3 4 3 4"
    print_linked_list(ll3, 1,1);

    //Summing the elements, should be 30
    //printf("Sum: %d\n", sum_linked_list(ll3));

    //Freeing the memory of ll3
    destroy_linked_list(ll3);
}
