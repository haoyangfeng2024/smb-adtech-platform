import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Plus, Search, Filter, MoreVertical } from 'lucide-react'

interface Campaign {
  id: string
  name: string
  status: 'active' | 'paused' | 'draft'
  budget: number
  spend: number
  impressions: number
  clicks: number
  ctr: number
}

const campaigns: Campaign[] = [
  { id: '1', name: 'Summer Sale 2024', status: 'active', budget: 10000, spend: 4500, impressions: 890000, clicks: 18900, ctr: 2.1 },
  { id: '2', name: 'Brand Awareness', status: 'active', budget: 5000, spend: 3200, impressions: 560000, clicks: 10080, ctr: 1.8 },
  { id: '3', name: 'Product Launch', status: 'paused', budget: 8000, spend: 2100, impressions: 320000, clicks: 4800, ctr: 1.5 },
  { id: '4', name: 'Retargeting', status: 'draft', budget: 3000, spend: 0, impressions: 0, clicks: 0, ctr: 0 },
]

export default function Campaigns() {
  const [searchTerm, setSearchTerm] = useState('')

  const filteredCampaigns = campaigns.filter(c => 
    c.name.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Campaigns</h1>
          <p className="mt-1 text-sm text-gray-500">Manage your advertising campaigns</p>
        </div>
        <Link
          to="/campaigns/new"
          className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Campaign
        </Link>
      </div>

      {/* Filters */}
      <div className="flex items-center space-x-4">
        <div className="relative flex-1 max-w-md">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <Search className="w-5 h-5 text-gray-400" />
          </div>
          <input
            type="text"
            className="block w-full py-2 pl-10 text-gray-900 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
            placeholder="Search campaigns..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50">
          <Filter className="w-4 h-4 mr-2" />
          Filter
        </button>
      </div>

      {/* Campaign List */}
      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Campaign</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Budget</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Spend</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Impressions</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Clicks</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">CTR</th>
              <th className="relative px-6 py-3"><span className="sr-only">Edit</span></th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredCampaigns.map((campaign) => (
              <tr key={campaign.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">{campaign.name}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                    campaign.status === 'active' ? 'bg-green-100 text-green-800' :
                    campaign.status === 'paused' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {campaign.status}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  ${campaign.budget.toLocaleString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  ${campaign.spend.toLocaleString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {campaign.impressions.toLocaleString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {campaign.clicks.toLocaleString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {campaign.ctr}%
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <button className="text-gray-400 hover:text-gray-500">
                    <MoreVertical className="w-5 h-5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
