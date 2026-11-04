interface Props { icon: string; title: string; sub: string }
export default function PageHeader({ icon, title, sub }: Props) {
  return (
    <div className="bg-gradient-to-r from-gray-900 to-gray-900/60 border border-gray-800
                    border-l-4 border-l-blue-600 rounded-xl p-5 mb-6">
      <h1 className="text-xl font-bold text-white flex items-center gap-2">
        <span>{icon}</span> {title}
      </h1>
      <p className="text-gray-400 text-sm mt-1">{sub}</p>
    </div>
  )
}
