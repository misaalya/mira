/* WaitCompositor - wait (up to 3 s) until a compositor owns _NET_WM_CM_S0.
 *
 * GTK and polybar pick their X visual when the window is created: without a
 * compositor they get a 24-bit visual and every transparent area is painted
 * black. bspwmrc runs this after starting picom and before Theme.sh starts
 * the bar. Exits 0 either way (no display: nothing to wait for). */
#include <X11/Xlib.h>
#include <time.h>

int main(void) {
    Display *d = XOpenDisplay(NULL);
    if (!d) return 0;
    Atom cm = XInternAtom(d, "_NET_WM_CM_S0", False);
    struct timespec step = {0, 100 * 1000 * 1000};
    for (int i = 0; i < 30 && XGetSelectionOwner(d, cm) == None; i++) nanosleep(&step, NULL);
    XCloseDisplay(d);
    return 0;
}
