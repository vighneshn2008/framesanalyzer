"""Entry point: python main.py  (GUI).  Use  python main.py --cli  for the text menu."""
import sys

if __name__ == "__main__":
    if "--cli" in sys.argv:
        import project
        project.main()
    else:
        from gui import RollingBodyApp
        RollingBodyApp().mainloop()
