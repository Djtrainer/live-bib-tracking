import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Formats time from milliseconds to a MM:SS.ms string.
 * Example: 321350 -> "05:21.35"
 */
export function formatTime(milliseconds: number): string {
    if (typeof milliseconds !== 'number' || isNaN(milliseconds)) {
        return "00:00.00";
    }

    const totalSeconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const centiseconds = Math.floor((milliseconds % 1000) / 10);

    const formattedMinutes = minutes.toString().padStart(2, '0');
    const formattedSeconds = seconds.toString().padStart(2, '0');
    const formattedCentiseconds = centiseconds.toString().padStart(2, '0');

    return `${formattedMinutes}:${formattedSeconds}.${formattedCentiseconds}`;
}