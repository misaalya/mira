/* A tiny JSON reader (enough for `bspc wm -d`): parse once, walk, free. */
#ifndef MISA_UI_JSON_H
#define MISA_UI_JSON_H

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct J {
    JType type;
    double num;           /* J_NUM, J_BOOL (0/1) */
    char *str;            /* J_STR */
    char *key;            /* member name inside an object */
    struct J *child;      /* first element / member */
    struct J *next;
} J;

J *json_parse(const char *text);
void json_free(J *j);
J *jget(const J *obj, const char *key);       /* NULL if missing */
const char *jstr(const J *j);                 /* "" unless a string */
double jnum(const J *j);                      /* 0 unless a number */
int jtrue(const J *j);

#endif
