import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { TrendingUp, TrendingDown, DollarSign, MousePointer, Eye, ArrowUpRight } from 'lucide-react'

interface Stats {
  totalSpend: number
  impressions: number
  clicks: number
  conversions: number
  ctr: number
  roas: number
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats>({
    totalSpend: 12450,
    impressions: 2456789,
    clicks: 45678,
    conversions: 2345,
    ctr: 1.86,
    roas: 3.2,
  })

  const metricCards = [
    { name: 'Total Spend', value: `$${stats.totalSpend.toLocaleString()}`, change: '+12%', trend: 'up', icon: DollarSign },
    { name: 'Impressions', value: (stats.impressions / 1000).toFixed(1) + 'K', change: '+8%', trend: 'up', icon: Eye },
    { name: 'Clicks', value: (stats.clicks / 1000).toFixed(1) + 'K', change: '+15%', trend: 'up', icon: MousePointer },
    { name: 'CTR', value: stats.ctr + '%', change: '+0.2%', trend: 'up', icon: TrendingUp },
    { name: 'ROAS', value: stats.roas + 'x', change: '+0.5x', trend: 'up', icon: TrendingUp },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="mt-1 text-sm text-gray-500">Overview of your advertising performance</p>
        </div>
        <Link
          to="/campaigns/new"
          className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700"
        >
          <ArrowUpRight className="w-4 h-4 mr-2" />
          New Campaign
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
        {metricCards.map((metric) => (
          <div key={metric.name} className="overflow-hidden bg-white rounded-lg shadow">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <metric.icon className="w-6 h-6 text-gray-400" />
                </div>
                <div className="flex-1 ml-3">
                  <dt className="text-sm font-medium text-gray-500 truncate">{metric.name}</dt>
                  <dd className="flex items-baseline">
                    <div className="text-2xl font-semibold text-gray-900">{metric.value}</div>
                    <div className={`ml-2 text-sm font-medium ${metric.trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                      {metric.change}
                    </div>
                  </dd>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Campaigns */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Active Campaigns</h3>
        </div>
        <div className="divide-y divide-gray-200">
          {[
            { name: 'Summer Sale 2024', status: 'Active', spend: 4500, ctr: 2.1 },
            { name: 'Brand Awareness', status: 'Active', spend: 3200, ctr: 1.8 },
            { name: 'Product Launch', status: 'Paused', spend: 2100, ctr: 1.5 },
          ].map((campaign) => (
            <div key={campaign.name} className="px-6 py-4 flex items-center justify-between hover:bg-gray-50">
              <div>
                <p className="text-sm font-medium text-gray-900">{campaign.name}</p>
                <p className="text-sm text-gray-500">{campaign.status}</p>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">${campaign.spend.toLocaleString()}</p>
                <p className="text-sm text-gray-500">CTR: {campaign.ctr}%</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
