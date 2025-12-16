import { useState } from "react";

interface DeleteAccountModalProps {
  username: string;
  onConfirm: () => Promise<void>;
  onCancel: () => void;
}

export default function DeleteAccountModal({ username, onConfirm, onCancel }: DeleteAccountModalProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState("");

  const handleConfirm = async () => {
    setIsDeleting(true);
    setError("");
    
    try {
      await onConfirm();
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Failed to delete account");
      setIsDeleting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 rounded-lg border border-slate-700 shadow-xl max-w-md w-full p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-full bg-red-950 flex items-center justify-center">
            <svg className="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-100">Delete Account?</h2>
            <p className="text-sm text-slate-400">This action cannot be undone</p>
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-4 mb-4">
          <p className="text-sm text-slate-300 mb-3">
            Are you sure you want to delete your account <span className="font-semibold text-slate-100">{username}</span>?
          </p>
          <p className="text-sm text-red-400">
            This will permanently delete:
          </p>
          <ul className="text-sm text-slate-300 mt-2 space-y-1 ml-4">
            <li>• Your login credentials</li>
            <li>• All uploaded documents</li>
            <li>• All embeddings and vector data</li>
            <li>• All critique logs and history</li>
          </ul>
        </div>

        {error && (
          <div className="bg-red-950 border border-red-800 rounded-lg p-3 mb-4">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        <div className="flex gap-3">
          <button
            onClick={onCancel}
            disabled={isDeleting}
            className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 text-slate-100 font-medium rounded-md transition"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={isDeleting}
            className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-800 text-white font-medium rounded-md transition"
          >
            {isDeleting ? "Deleting..." : "Yes, Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
