import { useState, useRef, useEffect } from "react";

interface UserDropdownProps {
  username: string;
  isGuest: boolean;
  onLogout: () => void;
  onDeleteAccount: () => void;
}

export default function UserDropdown({ username, isGuest, onLogout, onDeleteAccount }: UserDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  const handleStartOver = () => {
    onLogout();
    window.location.reload();
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 rounded-lg border border-emerald-500 px-3 py-1.5 text-xs sm:text-sm font-medium text-slate-100 hover:bg-slate-800"
      >
        <span className="hidden sm:inline">{username}</span>
        <span className="sm:hidden">{username}</span>
        <svg
          className={`w-4 h-4 text-emerald-500 transition-transform ${isOpen ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 rounded-lg border border-slate-700 bg-slate-900 shadow-xl z-50">
          <div className="p-2">
            <button
              onClick={() => {
                if (isGuest) {
                  handleStartOver();
                } else {
                  onLogout();
                  setIsOpen(false);
                }
              }}
              className="w-full text-left px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 hover:text-slate-100 rounded-md transition"
            >
              {isGuest ? "Start Over" : "Logout"}
            </button>
            
            {!isGuest && (
              <button
                onClick={() => {
                  onDeleteAccount();
                  setIsOpen(false);
                }}
                className="w-full text-left px-3 py-2 text-sm text-red-400 hover:bg-red-950 hover:text-red-300 rounded-md transition mt-1"
              >
                Delete Account
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
