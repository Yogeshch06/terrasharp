"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function NavLinks() {
  const pathname = usePathname();
  const links = [
    { href: "/", label: "Overview" },
    { href: "/enhance", label: "Enhance" },
    { href: "/change-detection", label: "Change detection" },
  ];

  return (
    <nav className="flex items-center gap-1">
      {links.map((link) => {
        const active = pathname === link.href;
        return (
          <Link
            key={link.href}
            href={link.href}
            className={`px-3 py-2 text-sm ${active ? "text-primary font-semibold" : "text-muted-foreground hover:text-foreground"}`}
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}