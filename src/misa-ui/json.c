/* A tiny recursive-descent JSON reader. Strict enough for bspc's output;
 * returns NULL on malformed input. */
#include "json.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *s; } P;

static void ws(P *p) { while (*p->s && isspace((unsigned char)*p->s)) p->s++; }

static J *node(JType t) {
    J *j = calloc(1, sizeof *j);
    if (j) j->type = t;
    return j;
}

static void put_utf8(char **o, unsigned c) {
    char *d = *o;
    if (c < 0x80) *d++ = c;
    else if (c < 0x800) { *d++ = 0xC0 | c >> 6; *d++ = 0x80 | (c & 63); }
    else if (c < 0x10000) { *d++ = 0xE0 | c >> 12; *d++ = 0x80 | (c >> 6 & 63); *d++ = 0x80 | (c & 63); }
    else { *d++ = 0xF0 | c >> 18; *d++ = 0x80 | (c >> 12 & 63); *d++ = 0x80 | (c >> 6 & 63); *d++ = 0x80 | (c & 63); }
    *o = d;
}

static unsigned hex4(const char *s) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return 0xFFFD;
    }
    return v;
}

static char *string(P *p) {
    if (*p->s != '"') return NULL;
    p->s++;
    const char *e = p->s;
    size_t n = 0;
    while (*e && *e != '"') { if (*e == '\\' && e[1]) e++; e++; n++; }
    if (*e != '"') return NULL;
    char *out = malloc(n * 4 + 1), *o = out;
    if (!out) return NULL;
    while (*p->s != '"') {
        char c = *p->s++;
        if (c != '\\') { *o++ = c; continue; }
        c = *p->s++;
        switch (c) {
            case 'n': *o++ = '\n'; break;
            case 't': *o++ = '\t'; break;
            case 'r': *o++ = '\r'; break;
            case 'b': *o++ = '\b'; break;
            case 'f': *o++ = '\f'; break;
            case 'u': {
                if (strlen(p->s) < 4) { free(out); return NULL; }
                unsigned u = hex4(p->s);
                p->s += 4;
                if (u >= 0xD800 && u < 0xDC00 && p->s[0] == '\\' && p->s[1] == 'u') {
                    unsigned lo = hex4(p->s + 2);
                    if (lo >= 0xDC00 && lo < 0xE000) {
                        u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                        p->s += 6;
                    }
                }
                put_utf8(&o, u);
                break;
            }
            default: *o++ = c;
        }
    }
    p->s++;
    *o = 0;
    return out;
}

static J *value(P *p, int depth);

static J *container(P *p, int depth, int obj) {
    J *j = node(obj ? J_OBJ : J_ARR), **tail = &j->child;
    p->s++;
    ws(p);
    if (*p->s == (obj ? '}' : ']')) { p->s++; return j; }
    for (;;) {
        char *key = NULL;
        ws(p);
        if (obj) {
            key = string(p);
            ws(p);
            if (!key || *p->s != ':') { free(key); json_free(j); return NULL; }
            p->s++;
        }
        J *v = value(p, depth + 1);
        if (!v) { free(key); json_free(j); return NULL; }
        v->key = key;
        *tail = v;
        tail = &v->next;
        ws(p);
        if (*p->s == ',') { p->s++; continue; }
        if (*p->s == (obj ? '}' : ']')) { p->s++; return j; }
        json_free(j);
        return NULL;
    }
}

static J *value(P *p, int depth) {
    if (depth > 256) return NULL;
    ws(p);
    char c = *p->s;
    if (c == '{') return container(p, depth, 1);
    if (c == '[') return container(p, depth, 0);
    if (c == '"') {
        char *s = string(p);
        if (!s) return NULL;
        J *j = node(J_STR);
        j->str = s;
        return j;
    }
    if (!strncmp(p->s, "true", 4)) { p->s += 4; J *j = node(J_BOOL); j->num = 1; return j; }
    if (!strncmp(p->s, "false", 5)) { p->s += 5; return node(J_BOOL); }
    if (!strncmp(p->s, "null", 4)) { p->s += 4; return node(J_NULL); }
    char *end;
    double d = strtod(p->s, &end);
    if (end == p->s) return NULL;
    p->s = end;
    J *j = node(J_NUM);
    j->num = d;
    return j;
}

J *json_parse(const char *text) {
    if (!text) return NULL;
    P p = {text};
    return value(&p, 0);
}

void json_free(J *j) {
    while (j) {
        J *n = j->next;
        json_free(j->child);
        free(j->str);
        free(j->key);
        free(j);
        j = n;
    }
}

J *jget(const J *obj, const char *key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    for (J *c = obj->child; c; c = c->next)
        if (c->key && !strcmp(c->key, key)) return c;
    return NULL;
}

const char *jstr(const J *j) { return j && j->type == J_STR ? j->str : ""; }
double jnum(const J *j) { return j && (j->type == J_NUM || j->type == J_BOOL) ? j->num : 0; }
int jtrue(const J *j) { return j && j->type == J_BOOL && j->num != 0; }
