import { cn } from "@/utils/tw";
import { BadgeDollarSign } from 'lucide-react';

export type LogoProps = {
  className?: string;
};

const Logo = ({ className }: LogoProps) => {
  return (
    <div className={cn("flex gap-2 text-xl items-center", className)} aria-describedby="Logo">
      <BadgeDollarSign className="h-5" />
      <span className="font-medium">Trade Master</span>
    </div>
  );
};

export default Logo;
